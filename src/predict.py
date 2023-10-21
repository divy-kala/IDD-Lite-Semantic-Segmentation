import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from models.unet import Net
from models.unet_l import UNet_l
from models.deeplabv3_resnet import DeepnetLabV3_ResNet
from models.custom_fcn import CustomFCN
from datamodules.idd_datamodule import IDDDataModule


def main():
    model_path = '/data/divy/idd20k_lite/idd/runs/lightning_logs/IDD/version_54/checkpoints/epoch=107-val_acc=0.52-val_loss=0.21.ckpt'
    model = CustomFCN.load_from_checkpoint(model_path)
    
    idd_datamodule =  IDDDataModule(None, None, '../idd20k_lite/leftImg8bit/val/*/*_image.jpg',
                                    batch_size=16, num_workers=24)
    idd_datamodule.setup()
    test_dataloader = idd_datamodule.test_dataloader()

    trainer = pl.Trainer(devices=[0])
    predictions = trainer.predict(model, dataloaders=test_dataloader)

if __name__ == '__main__':
    main()