import os

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import albumentations as A

from PIL import Image
from glob import glob
import numpy as np


class IDDataset(Dataset):
    def __init__(self, image_paths, transform=None, image_label_transform=None):
        super().__init__()
        
        self.image_paths = glob(image_paths) 
        self.label_paths = [p.replace('leftImg8bit', 'gtFine').replace(
                        '_image.jpg', '_label.png') for p in self.image_paths ]

        self.transform = transform
        self.image_label_transform = image_label_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # img = cv2.imread(self.img_path)[:,:,::-1]
        img = Image.open(img_path).convert('RGB')
        # open is a lazy operation; this function identifies the file, 
        # but the file remains open and the actual image data is not read 
        # from the file until you try to process the data

        label = Image.open(self.label_paths[idx])
        
        # Transformations work on numpy arrays
        img = np.array(img)
        label = np.array(label).astype('float32')
        
        # img and label will be converted to tensor and float (if needed), and the transforms will be sent via datamodule
        if self.transform:
            if isinstance(self.transform, A.core.composition.Compose):
                transformed = self.transform(image=img)
                img = transformed['image']
            else:
                img = self.transform(img)
        if self.image_label_transform:
            transformed = self.image_label_transform(image=img, label=label)
            img, label = transformed['image'], transformed['label']

        # img = img.transpose(0,2) 
        # label = label[None,:,:]
        
        return img, label


if __name__ == '__main__':
    dataset = IDDataset('../idd20k_lite/leftImg8bit/train/*/*_image.jpg')
    print(f"The dataset has {len(dataset)} images")
    print(dataset[0][0].shape)