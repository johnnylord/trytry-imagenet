import os
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.models.resnet import resnet18
from tensorboardX import SummaryWriter


__all__ = [ "ImageNetAgent" ]


class ImageNetAgent:

    def __init__(self, config):
        self.config = config

        # Training environment
        self.device = config['train']['device'] if torch.cuda.is_available() else "cpu"

        # Dataset
        transform = T.Compose([
                        T.Resize([config['dataset']['size'], config['dataset']['size']]),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]) ])
        dataset = ImageFolder(config['dataset']['root'], transform=transform)

        # Train and valid split
        train_samples = int(len(dataset)*config['dataset']['ratio'])
        valid_samples = len(dataset) - train_samples
        train_dataset, valid_dataset = random_split(dataset, [train_samples, valid_samples])

        # Dataloader
        self.train_loader = DataLoader(train_dataset,
                                batch_size=config['dataloader']['batch_size'],
                                num_workers=config['dataloader']['num_workers'],
                                shuffle=True)
        self.valid_loader = DataLoader(valid_dataset,
                                batch_size=config['dataloader']['batch_size'],
                                num_workers=config['dataloader']['num_workers'],
                                shuffle=False)

        # Model
        self.model = resnet18()
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=config['optim']['lr'])

        # Loss funciton
        self.criterion = nn.CrossEntropyLoss()

        # Tensorboard
        log_dir = osp.join(config['train']['log_dir'], config['train']['exp_name'])
        self.writer = SummaryWriter(logdir=log_dir)

        # Dynamic state
        self.current_epoch = -1
        self.current_loss = 10000

        # Resume training or not
        if config['train']['resume']:
            checkpoint_dir = osp.join(self.config['train']['log_dir'],
                                    "{}_checkpoint".format(self.config['train']['exp_name']))
            checkpoint_path = osp.join(checkpoint_dir, 'best.pth')
            checkpoint = torch.load(checkpoint_path)

            # Load pretrained model
            self.model = self.model.load_state_dict(checkpoint['model'])

            # Load optimier
            self.optimizer = self.optimizer.load_state_dict(checkpoint['optimizer'])

            # Resume to training state
            self.current_loss = checkpoint['current_loss']
            self.current_epoch = checkpoint['current_loss']

    def train(self):
        for epoch in range(self.current_epoch+1, self.config['train']['n_epochs']):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.validate()

    def train_one_epoch(self):
        running_loss = 0
        running_corrects = 0

        self.model.train()
        for batch_idx, (imgs, labels) in enumerate(self.train_loader):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            preds = torch.max(outputs.data, 1)[1]
            corrects = float(torch.sum(preds == labels.data))
            running_corrects += corrects
            running_loss += loss.item()*len(imgs)

            if batch_idx % self.config['train']['interval'] == 0:
                print("Epoch {}:{}({}%), Loss: {:.2f}".format(
                    self.current_epoch, self.config['train']['n_epochs'],
                    int(batch_idx*100/len(self.train_loader)), loss.item()))

        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = running_corrects / len(self.train_loader.dataset)
        self.writer.add_scalar("Train Loss", epoch_loss, self.current_epoch)
        self.writer.add_scalar("Train Acc", epoch_acc, self.current_epoch)
        print("Epoch {}:{}, Train Loss: {:.2f}, Train Acc: {:.2f}".format(
            self.current_epoch, self.config['train']['n_epochs'],
            epoch_loss, epoch_acc))

    def validate(self):
        running_loss = 0
        running_corrects = 0

        self.model.eval()
        for batch_idx, (imgs, labels) in enumerate(self.valid_loader):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)

            preds = torch.max(outputs.data, 1)[1]
            corrects = float(torch.sum(preds == labels.data))
            running_corrects += corrects
            running_loss += loss.item()*len(imgs)

        epoch_loss = running_loss / len(self.valid_loader.dataset)
        epoch_acc = running_corrects / len(self.valid_loader.dataset)
        self.writer.add_scalar("Valid Loss", epoch_loss, self.current_epoch)
        self.writer.add_scalar("Valid Acc", epoch_acc, self.current_epoch)
        print("Epoch {}:{}, Valid Loss: {:.2f}, Valid Acc: {:.2f}".format(
            self.current_epoch, self.config['train']['n_epochs'],
            epoch_loss, epoch_acc))

        if epoch_loss < self.current_loss:
            self.current_loss = epoch_loss
            self._save_checkpoint()

    def finalize(self):
        pass

    def _save_checkpoint(self):
        checkpoints = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'current_epoch': self.current_epoch,
                'current_loss': self.current_loss }

        checkpoint_dir = osp.join(self.config['train']['log_dir'],
                                "{}_checkpoint".format(self.config['train']['exp_name']))
        if not osp.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = osp.join(checkpoint_dir, 'best.pth')
        torch.save(checkpoints, checkpoint_path)
        print("Save checkpoint to '{}'".format(checkpoint_path))
