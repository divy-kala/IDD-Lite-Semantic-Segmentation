import os
from datetime import datetime
import warnings

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
import torch
import pytorch_lightning as pl
import torchvision.models as models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torch.optim.lr_scheduler import ReduceLROnPlateau
import PIL.Image as Image

# from focal_loss import sparse_categorical_focal_loss 

from src.utils.padding import pad_to, unpad
from src.utils.loss import DiceLoss


class CustomFCN(pl.LightningModule):

    def __init__(self, nc=3, n_classes=8, model_hparams=None) :
        # model_hparams - Hyperparameters for the model, as dictionary.
        super(CustomFCN, self).__init__()
        self.save_hyperparameters()
        self.test_step_outputs = {'loss':[], 'predicted labels':[], 'true labels':[]}        

        self.no_classes = n_classes
        self.nc = nc  # Number of input channels

        self.block1_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.block1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.block1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block2_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.block2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.block2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block3_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.block3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.block3_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.block3_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block4_conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.block4_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block4_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block4_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block5_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.pred3 = nn.ConvTranspose2d(256, n_classes, kernel_size=16, stride=16, bias=False)
        self.pred4 = nn.ConvTranspose2d(512, n_classes, kernel_size=32, stride=32, bias=False)
        self.pred5 = nn.ConvTranspose2d(512, n_classes, kernel_size=64, stride=64, bias=False)
        
        self.add = nn.ModuleList([self.pred3, self.pred4, self.pred5])
        
        self.final = nn.MaxPool2d(kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(n_classes, n_classes, kernel_size=1)
        
        nn.init.kaiming_normal_(self.final_conv.weight, mode='fan_out', nonlinearity='relu')

        self.softmax = nn.Softmax(dim=1)
        
        weights = [0.39, 5.21, 7.76, 1.46, 1.05, 0.53, 0.62, 0 ] #595.24] # 1/(no_classes * class_prob_in_gt)
        weights = torch.tensor(weights)
        self.loss_module =  nn.CrossEntropyLoss(weight=weights) 
        # self.loss_module = lambda x,y: sparse_categorical_focal_loss(x,y, axis=1, gamma=2)
        # self.loss_module = DiceLoss()

        
        
    def forward(self, imgs):
        # Pad so U-Net concatenation works
        # x, pads = pad_to(imgs,16)    
        x = imgs

        x1 = F.relu(self.block1_conv1(x))
        x1 = F.relu(self.block1_conv2(x1))
        x1 = self.block1_pool(x1)
        
        x2 = F.relu(self.block2_conv1(x1))
        x2 = F.relu(self.block2_conv2(x2))
        x2 = self.block2_pool(x2)
        
        x3 = F.relu(self.block3_conv1(x2))
        x3 = F.relu(self.block3_conv2(x3))
        x3 = F.relu(self.block3_conv3(x3))
        x3 = self.block3_pool(x3)
        
        x4 = F.relu(self.block4_conv1(x3))
        x4 = F.relu(self.block4_conv2(x4))
        x4 = F.relu(self.block4_conv3(x4))
        x4 = self.block4_pool(x4)
        
        x5 = F.relu(self.block5_conv1(x4))
        x5 = F.relu(self.block5_conv2(x5))
        x5 = F.relu(self.block5_conv3(x5))
        x5 = self.block5_pool(x5)
        
        pred3 = self.pred3(x3)
        pred4 = self.pred4(x4)
        pred5 = self.pred5(x5)
        
        add = pred3 + pred4 + pred5
        final = self.final(add)
        final = self.final_conv(final)
        
        output = self.softmax(final)
        
        # Unpad before outputting
        # output = unpad(output, pads)
        
        return output  

    def configure_optimizers(self, lr=1e-3):
        optimizer = Adam(self.parameters(), lr=lr) 
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=10, verbose=True),
            'monitor' : 'val_loss'
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):

        imgs, seg_map = batch
        out = self(imgs)

        loss = self.loss_module(out, seg_map)
        # loss = np.sum(self.loss_module(seg_map.cpu(), out.cpu()))
        self.log('train_loss', loss) #, sync_dist=True, )
        return loss


    def validation_step(self, batch, batch_idx):
        imgs, seg_map = batch
        out = self(imgs)
        predictions = self.softmax(out)
        predictions = np.argmax(predictions.cpu().numpy(), axis=1)
        predictions = predictions[:, np.newaxis, :, :] #TODO: remove if labels only have 3 dims
        predictions = torch.from_numpy(predictions).to(seg_map.device)

        accuracy = (predictions == seg_map).to(torch.float).mean()
        seg_map = seg_map.squeeze(1) #TODO: Might as well not load it from dataloader in 8,1,227,320 form, but 8,227,320 form
        loss = self.loss_module(out, seg_map)
        # loss = np.sum(self.loss_module(seg_map.cpu(), out.cpu()))

        self.log("val_acc", accuracy, prog_bar=True, sync_dist=True)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        # TODO: Make it same as validation step once that works
        imgs, seg_map = batch
        out = self(imgs)
        predictions = self.softmax(out)
        predictions = np.argmax(predictions.cpu().numpy(), axis=1)
        predictions = predictions[:, np.newaxis, :, :] #TODO: remove if labels only have 3 dims

        accuracy = (predictions == seg_map).to(torch.float).mean()
        loss = self.loss_module(out, seg_map)

        self.log('test_loss', loss, prog_bar=True, sync_dist=True )
        self.log("test_acc", accuracy, prog_bar=True, sync_dist=True)


    def predict_step(self, batch, batch_idx):
        if isinstance(batch, list):
            imgs, img_paths = batch
        else: imgs = batch
        out = self(imgs)
        predictions = self.softmax(out)
        predictions = np.argmax(predictions.cpu().numpy(), axis=1)
  

        for path in img_paths:
            dir_name = os.path.basename(os.path.dirname(path))
            file_name = os.path.basename(path).replace('_image.jpg', '_label.png')
            dst_dir = os.path.join('preds', dir_name)
            os.makedirs(dst_dir, exist_ok=True)
            full_path = os.path.join(dst_dir, file_name)
            img = Image.fromarray(predictions[0].astype('uint8'))
            img.save(full_path)

        # predictions = predictions[:, np.newaxis, :, :] #TODO: remove if labels only have 3 dims

        return predictions

    def predict(self, imgs):
        if len(imgs.shape) != 4:
            raise Exception("Please send batched input with batch in the first dimension.")
        elif imgs.shape[1] != 3:
            imgs = imgs.permute(0,3,1,2)
            warnings.warn("Please ensure the images are channels first -> (batch_size, 3, 640, 640)")
        with torch.no_grad():
            return self.predict_step(imgs, None)