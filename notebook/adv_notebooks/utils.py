import os
import torch
import random
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fix_seeds(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_model(model, name='model.pth'):
    state = {'state_dict': model.state_dict()}
    torch.save(state, os.path.join('./models/', name))

def load_model(model, name='model.pth'):
    state = torch.load(os.path.join('./models/', name))
    model.load_state_dict(state['state_dict'])

def defense_noise(maginitude):
    def defense_transform(x):
        # noise = torch.FloatTensor(x.shape).uniform_(-maginitude, maginitude) #uniform
        noise = maginitude * torch.sign(torch.rand(x.shape))

        return torch.clamp(x + noise, min=0, max=1)
    return defense_transform

def get_train_transform():
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    return transform

# def svd(svd_ratio):
#     def svd_transform(x):
#         u, s, v = torch.svd(x)
#         num_singulars = int(s.size(-1) * svd_ratio)
#         s = s[..., :num_singulars]

#         res = torch.zeros_like(x)
#         for i in range(x.size(0)):
#             res[i] = u[i, :, :num_singulars] @ torch.diag(s[i, :num_singulars]) @ v[i, :num_singulars, :]
#         save_image(res, './plots/img.png')
#         return res
#     return svd_transform

import numpy as np
import math
from PIL import Image
def svd(svd_ratio):
    def svd_transform(x):
        x.save('./plots/adv.png')
        x = np.array(x)
        res = np.zeros_like(x)
        for i in range(3):
            u, s, v = np.linalg.svd(x[..., i])
            size = s.size
            n_singular = math.floor(size * svd_ratio)
            res[..., i] = u[:, :n_singular] @ np.diag(s[: n_singular]) @ v[: n_singular, :]
        res = Image.fromarray(res)
        res.save('./plots/svd.png')
        return res
    return svd_transform

def get_test_transform(svd_ratio=None, down_up_sample=None, filter_window_size=None):
    svd_transform = lambda x: x

    if svd_ratio is not None:
        svd_transform = svd(svd_ratio=svd_ratio)

    if down_up_sample is not None:
        pass

    if filter_window_size is not None:
        pass

    transform = transforms.Compose([
        svd_transform,
        transforms.ToTensor(),
    ])

    return transform

def save_adversarial_example(adv_img, mode, attack, category, fname):
    folder = os.path.join('./adversarial_examples', mode, attack, category)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, fname)
    save_image(adv_img.squeeze(0), path)

if __name__ == '__main__':
    adv_img = torch.rand(1, 3, 28, 28)
    save_adversarial_example(adv_img, 'cifar10', 'fgsm', 'airplane', '0001.png')