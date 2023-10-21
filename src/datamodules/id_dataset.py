import os

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import albumentations as A

from PIL import Image
from glob import glob
import numpy as np


class IDDataset(Dataset):
    def __init__(self, image_paths, transform=None, image_label_transform=None, return_file_paths = False, load_labels=True):
        super().__init__()
        
        self.image_paths = glob(image_paths) 
        self.load_labels = load_labels
        if load_labels:
            self.label_paths = [p.replace('leftImg8bit', 'gtFine').replace(
                        '_image.jpg', '_label.png') for p in self.image_paths ]

        self.transform = transform
        self.image_label_transform = image_label_transform
        self.return_file_paths = return_file_paths
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # img = cv2.imread(self.img_path)[:,:,::-1]
        img = Image.open(img_path).convert('RGB')
        # open is a lazy operation; this function identifies the file, 
        # but the file remains open and the actual image data is not read 
        # from the file until you try to process the data

        if self.load_labels:
            label = Image.open(self.label_paths[idx])
            label = label.resize((256,256), resample=Image.NEAREST)
            label = np.array(label).astype('long')
        
        # Transformations work on numpy arrays
        img = img.resize((256,256), resample=Image.NEAREST)
        img = np.array(img)
        
        
        # # img and label will be converted to tensor and float (if needed), and the transforms will be sent via datamodule
        # if self.transform:
        #     if isinstance(self.transform, A.core.composition.Compose):
        #         transformed = self.transform(image=img)
        #         img = transformed['image']
        #     else:
        #         img = self.transform(img)
        # if self.load_labels and self.image_label_transform:
        #     transformed = self.image_label_transform(image=img, label=label.astype('uint8')) #OpenCV operations in ShiftScaleRotate start throwing errors because they don't work with longs
        #     img, label = transformed['image'], transformed['label']
            

        if self.load_labels:
            label[label == 255] = 7
            label = label.astype('long') #long()
            if len(label.shape) == 3: label = label.squeeze(0)  # Convert 1HW to HW so convolution happen without error
        
        if self.return_file_paths:
            if self.load_labels:
                return img, label, self.image_paths[idx]   
            else:
                return img, self.image_paths[idx]          


        return img.transpose(2,0,1).astype('float32')/255, label


if __name__ == '__main__':
    # dataset = IDDataset('../idd20k_lite/leftImg8bit/train/*/*_image.jpg')
    dataset = IDDataset('/data/divy/idd20k_lite/full_dataset/small_idd20kII/leftImg8bit/train/*/*.jpg')
    print(f"The dataset has {len(dataset)} images")
    print(dataset[0][0].shape)
    

