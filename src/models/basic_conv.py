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
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smpu
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss
import torchvision
import cv2
from random import randrange

class BasicConv(pl.LightningModule):
    def __init__(self, num_classes=8, encoder_name='efficientnet-b2'):
        super(BasicConv, self).__init__()
        
        nc = 3
        ndf = 4
        # Code borrowed from DCGAN
        self.enc = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )
        nz = ndf*8
        ngf = ndf
        self.dec = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf,      8, 4, 2, 1, bias=False)
            # state size. (nc) x 64 x 64
        )
        
        self.softmax = nn.Softmax(dim=1)
        
        # weights = [0.39, 5.21, 7.76, 1.46, 1.05, 0.53, 0.62, 0 ] #595.24] # 1/(no_classes * class_prob_in_gt)
        # weights = torch.tensor(weights)
        weights = None
        self.loss_module =  nn.CrossEntropyLoss(weight=weights) 
        self.count = 0
        
    
        # self.loss_module2 = DiceLoss('multiclass', [0,1,2,3,4,5,6])
        # self.iouLoss = JaccardLoss('multiclass', [0,1,2,3,4,5,6])


    def forward(self, x):
        x = self.enc(x)
        # print(x.shape)

        x = self.dec(x)
        # print(x.shape)
        output = x#[:, :, 8:328, 6:233]
        return output

    def configure_optimizers(self, lr=1e-4):
        optimizer = Adam(self.parameters(), lr=lr) 
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=10, verbose=True),
            'monitor' : 'train_loss'
        }
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):

        imgs, seg_map = batch
        out = self(imgs)

        loss = self.loss_module(out, seg_map)
        # loss += self.loss_module2(out,seg_map)
        # loss += self.iouLoss(out, seg_map.squeeze(1).long() )
        # loss = np.sum(self.loss_module(seg_map.cpu(), out.cpu()))
        self.log('train_loss', loss) #, sync_dist=True, )

        if self.count >= 100:
            self.count = 0
            predictions = self.softmax(out)
            predictions = np.argmax(predictions.detach().cpu().numpy(), axis=1)
            predictions = predictions[:, np.newaxis, :, :] #TODO: remove if labels only have 3 dims
            predictions = torch.from_numpy(predictions).to(seg_map.device)

            accuracy = (predictions == seg_map).to(torch.float).mean()
            iou = smpu.metrics.IoU()(predictions, seg_map)
            loss = self.loss_module(out, seg_map)
            # loss += self.loss_module2(out,seg_map)
            # loss = self.iouLoss(out, seg_map.squeeze(1).long() )
            
            # Squeezing might be needeed with BCE and other losses
            seg_map = seg_map.squeeze(1)  #TODO: Might as well not load it from dataloader in 8,1,227,320 form, but 8,227,320 form
            
            # loss = np.sum(self.loss_module(seg_map.cpu(), out.cpu()))

            self.log("train_acc", accuracy, prog_bar=True, sync_dist=True)
            self.log('train_iou', iou, prog_bar=True, sync_dist=True)
            
            
            i = randrange(imgs.shape[0])

            cv2.imwrite("img_train.jpg", (imgs*255).byte()[i].detach().cpu().numpy().transpose(1,2,0)[:,:,::-1])
            cv2.imwrite("label_train.jpg", (seg_map.unsqueeze(1)[i]*36).detach().cpu().numpy().transpose(1,2,0)[:,:,::-1])
            cv2.imwrite("preds_train.jpg", (predictions[i]*36).detach().cpu().numpy().transpose(1,2,0)[:,:,::-1])

        self.count += 1
        return loss


    def validation_step(self, batch, batch_idx):
        imgs, seg_map = batch
        out = self(imgs)
        predictions = self.softmax(out)
        predictions = np.argmax(predictions.cpu().numpy(), axis=1)
        predictions = predictions[:, np.newaxis, :, :] #TODO: remove if labels only have 3 dims
        predictions = torch.from_numpy(predictions).to(seg_map.device)

        accuracy = (predictions == seg_map).to(torch.float).mean()
        iou = smpu.metrics.IoU()(predictions, seg_map)
        loss = self.loss_module(out, seg_map)
        # loss += self.loss_module2(out,seg_map)
        # loss = self.iouLoss(out, seg_map.squeeze(1).long() )
        
        # Squeezing might be needeed with BCE and other losses
        seg_map = seg_map.squeeze(1)  #TODO: Might as well not load it from dataloader in 8,1,227,320 form, but 8,227,320 form
        
        # loss = np.sum(self.loss_module(seg_map.cpu(), out.cpu()))

        self.log("val_acc", accuracy, prog_bar=True, sync_dist=True)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_iou', iou, prog_bar=True, sync_dist=True)
        
        
        i = randrange(imgs.shape[0])

        cv2.imwrite("img.jpg", (imgs*255).byte()[i].cpu().numpy().transpose(1,2,0)[:,:,::-1])
        cv2.imwrite("label.jpg", (seg_map.unsqueeze(1)[i]*36).cpu().numpy().transpose(1,2,0)[:,:,::-1])
        cv2.imwrite("preds.jpg", (predictions[i]*36).cpu().numpy().transpose(1,2,0)[:,:,::-1])

        # grid_pred_labels = torchvision.utils.make_grid(predictions)
        # grid_images = torchvision.utils.make_grid(imgs)   
        # grid_labels = torchvision.utils.make_grid(seg_map.unsqueeze(1)) 

        # self.logger.experiment.add_image("generated_labels", grid_pred_labels, self.current_epoch)
        # self.logger.experiment.add_image("real_images", grid_images, self.current_epoch)
        # self.logger.experiment.add_image("true_labels", grid_labels, self.current_epoch)



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


