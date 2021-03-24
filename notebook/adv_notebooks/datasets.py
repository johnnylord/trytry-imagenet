import os
from PIL import Image
from glob import glob
from torch.utils.data import Dataset

label_idx = {}
with open('./label_to_idx.txt', mode='r') as f:
    for line in f.readlines():
        splits = line.split()
        label_idx[splits[0]] = int(splits[1])

def label_to_idx(labels):
    result = []
    for item in labels:
        label = item.split('/')[-1]
        result.append(label_idx[label])

    return result
    

class ImageDataset(Dataset):

    def __init__(self, folder, transform=None, return_details=False):
        self.transform = transform
        self.return_details = return_details
        self.images = []
        self.labels = []

        classes = sorted(glob(os.path.join(folder, '*')))
        label_idx = label_to_idx(classes)

        for i, image_folder in enumerate(classes):
            for image in sorted(glob(os.path.join(image_folder, '*.JPEG'))):
                self.images.append(image)
                self.labels.append(label_idx[i])
        

    def __getitem__(self, index):
        path = self.images[index]
        image = Image.open(path).convert('RGB')
        label = self.labels[index]
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.return_details:
            splits = path.split('/')
            category = splits[-2]
            fname = splits[-1]
            return image, label, category, fname
        
        return image, label
        
    def __len__(self):
        return len(self.labels)

class AdvDataset(Dataset):
    def __init__(self, adv_examples, transform=None):
        self.adv_examples = adv_examples
        self.transform = transform

    def __getitem__(self, index):
        if self.transform is not None:
            return self.transform(self.adv_examples[index])
        return self.adv_examples[index].cpu()

    def __len__(self):
        return len(self.adv_examples)

if __name__ == '__main__':
    folder = '/home/advattack/imagenet/ILSVRC_KAGGLE/sampled_imagenet'
    dataset = ImageDataset(folder, return_details=True)
    print(dataset.labels)