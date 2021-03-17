import os
import os.path as osp
import xml.etree.ElementTree as ET

from PIL import Image
from torch.utils.data import Dataset


__all__ = [ "PASCALImageNet" ]


class PASCALImageNet(Dataset):

    def __init__(self, img_dir, label_dir, class_to_idx, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.class_to_idx = class_to_idx
        self.transform = transform
        # Extract target file name
        self.fnames = [ osp.join(img_dir, f) for f in os.listdir(img_dir) ]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        bname = osp.basename(fname).split('.')[0]
        aname = osp.join(self.label_dir, bname+'.xml')

        img = Image.open(fname)
        if self.transform:
            img = self.transform(img)

        tree = ET.parse(aname)
        names = [ obj.find('name').text for obj in tree.findall('object') ]
        labels = [ self.class_to_idx[name] for name in names ]

        return img, labels[0]
