import os
import os.path as osp
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from tensorboardX import SummaryWriter

from model import get_model_cls


__all__ = [ "ImageNetAgent" ]


class ImageNetAgent:

    def __init__(self, config, rank=-1):
        self.rank = rank
        self.config = config

        # Training environment
        if config['train']['mode'] == 'parallel':
            gpu_id = config['train']['gpus'][rank]
            self.device = "cuda:{}".format(gpu_id)
        else:
            self.device = config['train']['device'] if torch.cuda.is_available() else "cpu"

        # Dataset
        train_transform = T.Compose([
                            T.RandomResizedCrop((config['dataset']['size'], config['dataset']['size'])),
                            T.RandomHorizontalFlip(),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]) ])
        valid_transform = T.Compose([
                            T.Resize(256),
                            T.CenterCrop((config['dataset']['size'], config['dataset']['size'])),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]) ])
        train_dataset = ImageFolder(config['dataset']['train']['root'], transform=train_transform)
        valid_dataset = ImageFolder(config['dataset']['valid']['root'], transform=valid_transform)

        # Dataloader
        if config['train']['mode'] == 'parallel':
            world_size = len(self.config['train']['gpus'])
            self.sampler = DistributedSampler(train_dataset)
            self.train_loader = DataLoader(train_dataset,
                                        sampler=self.sampler,
                                        batch_size=config['dataloader']['batch_size'],
                                        num_workers=config['dataloader']['num_workers'],
                                        pin_memory=True,
                                        shuffle=False)
        else:
            self.train_loader = DataLoader(train_dataset,
                                    batch_size=config['dataloader']['batch_size'],
                                    num_workers=config['dataloader']['num_workers'],
                                    pin_memory=True,
                                    shuffle=True)

        self.valid_loader = DataLoader(valid_dataset,
                                batch_size=config['dataloader']['batch_size'],
                                num_workers=config['dataloader']['num_workers'],
                                pin_memory=True,
                                shuffle=False)
        # Model
        model_cls = get_model_cls(config['model']['name'])
        model = model_cls(in_channels=config['model']['in_channels'],
                        num_classes=config['model']['num_classes'])
        if config['train']['mode'] == 'parallel':
            model = model.to(self.device)
            self.model = DDP(model,
                            device_ids=[config['train']['gpus'][rank]])
        else:
            self.model = model.to(self.device)

        # Optimizer
        self.optimizer = optim.SGD(self.model.parameters(),
                                lr=config['optimizer']['lr'],
                                momentum=config['optimizer']['momentum'],
                                weight_decay=config['optimizer']['weight_decay'])
        # Scheduler
        n_epochs = config['train']['n_epochs'] - config['scheduler']['warmup_epochs']
        def lr_lambda(epoch):
            return ((n_epochs-epoch)/n_epochs)**3
        scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1,
                                                total_epoch=config['scheduler']['warmup_epochs'],
                                                after_scheduler=scheduler)
        # Loss funciton
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        # Tensorboard
        if (
            (self.rank == 0 and config['train']['mode'] == 'parallel')
            or self.rank < 0
        ):
            log_dir = osp.join(config['train']['log_dir'], config['train']['exp_name'])
            self.writer = SummaryWriter(logdir=log_dir)

        # Dynamic state
        self.current_epoch = -1
        self.current_loss = 10000

    def resume(self):
        checkpoint_dir = osp.join(self.config['train']['log_dir'],
                                "{}_checkpoint".format(self.config['train']['exp_name']))
        checkpoint_path = osp.join(checkpoint_dir, 'best.pth')

        if config['train']['mode'] == 'parallel':
            master_gpu_id = config['train']['gpus'][0]
            map_location = { 'cuda:{}'.format(master_gpu_id): self.device }
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
        else:
            checkpoint = torch.load(checkpoint_path)

        # Load pretrained model
        self.model = self.model.load_state_dict(checkpoint['model'])

        # Load optimier
        self.optimizer = self.optimizer.load_state_dict(checkpoint['optimizer'])

        # Resume to training state
        self.current_loss = checkpoint['current_loss']
        self.current_epoch = checkpoint['current_epoch']
        print("Resume Training at epoch {}".format(self.current_epoch))

    def train(self):
        for epoch in range(self.current_epoch+1, self.config['train']['n_epochs']):
            self.current_epoch = epoch
            if self.config['train']['mode'] == 'parallel':
                self.sampler.set_epoch(self.current_epoch)
                self.train_one_epoch()
                self.validate()
                self.scheduler.step()
            else:
                self.train_one_epoch()
                self.validate()
                self.scheduler.step()

    def train_one_epoch(self):
        accs = []
        losses = []
        self.model.train()
        loop = tqdm(self.train_loader,
                desc=(
                    f"[{self.rank}] Train Epoch {self.current_epoch}/{self.config['train']['n_epochs']}"
                    f"- LR: {self.optimizer.param_groups[0]['lr']:.3f}"
                    ),
                leave=True)
        for batch_idx, (imgs, labels) in enumerate(loop):
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            preds = torch.max(outputs.data, 1)[1]
            corrects = float(torch.sum(preds == labels.data))
            acc = corrects/len(imgs)

            accs.append(acc)
            losses.append(loss.item())
            loop.set_postfix(loss=sum(losses)/len(losses),
                            acc=sum(accs)/len(accs))

        if self.rank <= 0:
            epoch_loss = sum(losses)/len(losses)
            epoch_acc = sum(accs)/len(accs)
            self.writer.add_scalar("Train Loss", epoch_loss, self.current_epoch)
            self.writer.add_scalar("Train Acc", epoch_acc, self.current_epoch)
            print("Epoch {}:{}, Train Loss: {:.2f}, Train Acc: {:.2f}".format(
                self.current_epoch, self.config['train']['n_epochs'],
                epoch_loss, epoch_acc))

    def validate(self):
        accs = []
        losses = []
        self.model.eval()
        loop = tqdm(self.valid_loader,
                desc=(
                    f"Valid Epoch {self.current_epoch}/{self.config['train']['n_epochs']}"
                    f"- LR: {self.optimizer.param_groups[0]['lr']:.3f}"
                    ),
                leave=True)
        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(loop):
                imgs = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)

                preds = torch.max(outputs.data, 1)[1]
                corrects = float(torch.sum(preds == labels.data))
                acc = corrects/len(imgs)

                accs.append(acc)
                losses.append(loss.item())
                loop.set_postfix(loss=sum(losses)/len(losses),
                                acc=sum(accs)/len(accs))

        epoch_loss = sum(losses)/len(losses)
        epoch_acc = sum(accs)/len(accs)
        print("Epoch {}:{}, Valid Loss: {:.2f}, Valid Acc: {:.2f}".format(
            self.current_epoch, self.config['train']['n_epochs'],
            epoch_loss, epoch_acc))
        if self.rank <= 0:
            self.writer.add_scalar("Valid Loss", epoch_loss, self.current_epoch)
            self.writer.add_scalar("Valid Acc", epoch_acc, self.current_epoch)
        if epoch_loss < self.current_loss:
            self.current_loss = epoch_loss
            if self.rank <= 0:
                self._save_checkpoint()

    def finalize(self):
        pass

    def _save_checkpoint(self):
        checkpoints = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'current_epoch': self.current_epoch,
                'current_loss': self.current_loss
                }
        checkpoint_dir = osp.join(self.config['train']['log_dir'], self.config['train']['exp_name'])
        if not osp.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = osp.join(checkpoint_dir, 'best.pth')
        torch.save(checkpoints, checkpoint_path)
        print("Save checkpoint to '{}'".format(checkpoint_path))
