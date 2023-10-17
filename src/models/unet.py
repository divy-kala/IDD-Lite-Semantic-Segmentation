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

from src.utils.padding import pad_to, unpad

class Net(pl.LightningModule):

    def __init__(self, nc=3, no_classes=8, filters=64, model_hparams=None) :
        # model_hparams - Hyperparameters for the model, as dictionary.
        super(Net, self).__init__()
        self.save_hyperparameters()
        self.test_step_outputs = {'loss':[], 'predicted labels':[], 'true labels':[]}        

        self.no_classes = no_classes
        self.nc = nc  # Number of input channels
        ndf = filters  # Number of encoder filters
        ngf = filters  # Number of decoder filters  

        self.loss_module =  nn.CrossEntropyLoss() 
        self.softmax = torch.nn.Softmax(dim=1)

        self.enc1 = nn.Sequential(
            nn.Conv2d(nc, ndf, 3, 2, bias=False, padding=1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),

        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 3, 2, bias=False, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, bias=False, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 3, 2, bias=False, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            # nn.ConvTranspose2d(ndf * 8, ngf * 8, 3, 1, bias=False, padding=0),
            # nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, bias=False, padding=1, output_padding=1),
            nn.ReLU(True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 2, 3, 2, bias=False, padding=1, output_padding=1),
            nn.ReLU(True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf, 3, 2, bias=False, padding=1, output_padding=1),
            nn.ReLU(True)
        )
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, ngf, 3, 2, bias=False, padding=1, output_padding=1),
            nn.ConvTranspose2d(ngf, no_classes, 3, 1, bias=False, padding=1),
        )
        
         
        
    def forward(self, imgs):
        # Pad so U-Net concatenation works
        x, pads = pad_to(imgs,16)    

        # Encode
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Decode with skip connections
        dec1 = self.dec1(enc4)
        dec2 = self.dec2(torch.cat([dec1, enc3], 1))
        dec3 = self.dec3(torch.cat([dec2, enc2], 1))
        output = self.final(torch.cat([dec3, enc1], 1))
        

        # dec2 = dec1 + enc3
        # dec3 = dec2 + enc2
        # output = dec3 + enc1

        # Unpad before outputting
        output = unpad(output, pads)
        
        return output  

    def configure_optimizers(self), lr=1e-2:
        optimizer = Adam(self.parameters(), lr=lr) 
        return optimizer

    def training_step(self, batch, batch_idx):

        imgs, seg_map = batch
        out = self(imgs)

        loss = self.loss_module(out, seg_map)
        self.log('train_loss', loss) #, sync_dist=True, )
        return loss


    def validation_step(self, batch, batch_idx):
        imgs, seg_map = batch
        out = self(imgs)
        predictions = self.softmax(out)
        predictions = np.argmax(predictions, axis=1)
        predictions = predictions[:, np.newaxis, :, :] #TODO: remove if labels only have 3 dims

        accuracy = (predictions == seg_map).to(torch.float).mean()
        loss = self.loss_module(out, seg_map)

        self.log("val_acc", accuracy, prog_bar=True, sync_dist=True)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        imgs, seg_map = batch
        out = self(imgs)
        predictions = self.softmax(out)
        predictions = np.argmax(predictions, axis=1)
        predictions = predictions[:, np.newaxis, :, :] #TODO: remove if labels only have 3 dims

        accuracy = (predictions == seg_map).to(torch.float).mean()
        loss = self.loss_module(out, seg_map)

        self.log('test_loss', loss, prog_bar=True, sync_dist=True )
        self.log("test_acc", accuracy, prog_bar=True, sync_dist=True)


    def predict_step(self, batch, batch_idx):
        imgs = batch
        out = self(imgs)
        predictions = self.softmax(out)
        predictions = np.argmax(predictions, axis=1)
        predictions = predictions[:, np.newaxis, :, :] #TODO: remove if labels only have 3 dims
        return predictions

    def predict(self, imgs):
        if len(imgs.shape) != 4:
            raise Exception("Please send batched input with batch in the first dimension.")
        elif imgs.shape[1] != 3:
            imgs = imgs.permute(0,3,1,2)
            warnings.warn("Please ensure the images are channels first -> (batch_size, 3, 640, 640)")
        with torch.no_grad():
            return self.predict_step(imgs, None)