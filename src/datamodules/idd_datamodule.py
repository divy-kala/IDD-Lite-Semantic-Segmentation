import os

from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
import albumentations as A
from albumentations.pytorch.transforms import ToTensor, ToTensorV2

from src.datamodules.id_dataset import IDDataset
import cv2

class IDDDataModule(LightningDataModule):
    def __init__(self, train_images_path: str=None, val_images_path: str=None, test_images_path: str=None,
                num_workers: int=2, batch_size: int = 8):
        super().__init__()
        self.batch_size = batch_size
        self.train_images_path = train_images_path
        self.val_images_path = val_images_path
        self.test_images_path = test_images_path
        self.num_workers = num_workers

    def setup(self, stage: str=None):

        # Not using Torch's transforms because similar transform also has to be applied on labels image, and torch is not elegant here
        # train_transform = transforms.Compose([
        # #     transforms.Resize(size=(640,640)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomRotation(20),
        #     transforms.ToTensor()
        #     #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])

        train_transform_image_and_label = A.Compose(
                        transforms=[
                                A.ShiftScaleRotate(interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0), mask_value=7), 
                                # Will ShiftScaleRotate labelmap with interpolated values if interpolation != 1 (nearest)
                                A.HorizontalFlip(p=0.5),
                                ToTensorV2(transpose_mask = False), # numpy HWC image is converted to pytorch CHW tensor and HW to 1HW
                                
                        ],
                        additional_targets={'label': 'image'}
                )
        
        train_transform_image = A.Compose(
                        transforms=[                      
                                A.RandomBrightnessContrast(p=0.2),
                                A.OneOf(
                                    [
                                        A.MotionBlur(blur_limit=3, p=1),
                                        A.GaussianBlur(blur_limit=3, p=1),
                                    ], p=0.5
                                ),
                                A.Perspective(p=0.5),
                                A.GaussNoise(p=0.5),
                                A.Sharpen(p=0.5),
                                # A.RandomSunFlare(),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                #Mean and std from ImageNet as they transfer reasonably
                                ]
                )
       

        # test_transform = transforms.Compose([
        # #     transforms.Resize(size=(640,640)),
        #     transforms.ToTensor(),
        #     # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to 
        #     # a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        
        val_transform = test_transform = transforms.Compose([
        #     transforms.Resize(size=(640,640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        
        self.train_dataset = IDDataset(self.train_images_path, transform=None, image_label_transform=None) \
                                 if self.train_images_path else None
        self.val_dataset = IDDataset(self.val_images_path, transform=None)if self.val_images_path else None
        self.test_dataset = IDDataset(self.test_images_path,transform=None, return_file_paths=True, load_labels=False)if self.test_images_path else None

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False)

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        pass


if __name__ == '__main__':
    idd_datamodule =  IDDDataModule('../idd20k_lite/leftImg8bit/train/*/*_image.jpg', '../idd20k_lite/leftImg8bit/val/*/*_image.jpg', 
                                    batch_size=256)
    idd_datamodule.setup()
    train_dl = idd_datamodule.train_dataloader()
    a = next(iter(train_dl))
    import numpy as np
    print(a[0].shape, a[1].shape, np.unique(a[1]))

    # test_dl = idd_datamodule.test_dataloader()
    # a = next(iter(test_dl))
    # print(a[0].shape, a[1].shape)
    
    val_dl = idd_datamodule.val_dataloader()
    b = next(iter(val_dl))
    print(b[0].shape, b[1].shape)
    
