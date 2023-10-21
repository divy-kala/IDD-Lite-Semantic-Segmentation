import os
from datetime import datetime
import warnings

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Conv2d, ReLU, ConvTranspose2d, MaxPool2d
from torch.optim import SGD, Adam
import torch
import pytorch_lightning as pl
import torchvision.models as models

from src.utils.padding import pad_to, unpad
from src.utils.loss import DiceLoss


class Block(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = Conv2d(inChannels, outChannels, 3, padding='same')
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, 3, padding='same')
        self.bn = nn.BatchNorm2d(outChannels)
        self.relu2 = ReLU()

    def forward(self, x):
        out = self.conv2(self.relu(self.conv1(x)))
        out = self.relu2(self.bn(out))
        return out
     

class Encoder(Module):
    def __init__(self, channels=(3, 16, 32, 64)):
        super().__init__()
        self.encBlocks = ModuleList(
            [Block(channels[i], channels[i + 1])
                 for i in range(len(channels) - 1)])
        self.pool = MaxPool2d(2)
    def forward(self, x):
        blockOutputs = []
    
        for block in self.encBlocks:
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)
        return blockOutputs


class Decoder(Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        # initialize the number of channels, upsampler blocks, and
        # decoder blocks
        self.channels = channels
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i + 1], 3, 2, padding=1, output_padding=1)
                 for i in range(len(channels) - 1)])
        self.dec_blocks = ModuleList(
            [Block(channels[i], channels[i + 1])
                 for i in range(len(channels) - 1)])
    def forward(self, x, encFeatures):
        for i in range(len(self.channels) - 1):
            x = self.upconvs[i](x)
            encFeat = encFeatures[i] # self.crop(encFeatures[i], x) 
            # encFeatures is a reversed list starting from second last encoded feature
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)
        return x
    # def crop(self, encFeatures, x):
    #     # grab the dimensions of the inputs, and crop the encoder
    #     # features to match the dimensions
    #     (_, _, H, W) = x.shape
    #     encFeatures = CenterCrop([H, W])(encFeatures)
    #     # return the cropped features
    #     return encFeatures


class UNet_l(pl.LightningModule):
    def __init__(self, encChannels=(3, 16, 32, 64),
         decChannels=(64, 32, 16),
         nbClasses=8, retainDim=True,
         outSize=(227, 320)):
        super().__init__()
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)
        self.head = Conv2d(decChannels[-1], nbClasses, 1)
        self.retainDim = retainDim
        self.outSize = outSize
        
        self.loss_module =  nn.CrossEntropyLoss() 
        # self.loss_module = DiceLoss()
        self.softmax = torch.nn.Softmax(dim=1)
          
    def forward(self, x):
        # x, pads = pad_to(x,32)          

        encFeatures = self.encoder(x)
        # Send the last encoded feature and the (reversed) list of encoded features (except the last one) starting from second last
        decFeatures = self.decoder(encFeatures[::-1][0],
            encFeatures[::-1][1:])
        # obtain the segmentation mask
        map = self.head(decFeatures)
        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them 
        # by default used 'nearest' algorithm, so no changes in label values

        # map = unpad(map, pads)

        if self.retainDim:
            map = F.interpolate(map, self.outSize)
        return map


    def configure_optimizers(self, lr=1e-3):
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
        predictions = np.argmax(predictions.cpu().numpy(), axis=1)
        predictions = predictions[:, np.newaxis, :, :] #TODO: remove if labels only have 3 dims
        predictions = torch.from_numpy(predictions).to(seg_map.device)

        accuracy = (predictions == seg_map).to(torch.float).mean()
        seg_map = seg_map.squeeze(1) #TODO: Might as well not load it from dataloader in 8,1,227,320 form, but 8,227,320 form
        loss = self.loss_module(out, seg_map)

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