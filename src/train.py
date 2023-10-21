import pytorch_lightning as pl
from models.unet import Net
from models.unet_l import UNet_l
from models.deeplabv3_resnet import DeepnetLabV3_ResNet
from models.custom_fcn import CustomFCN
from models.efficientnet import EfficientFCN
from datamodules.idd_datamodule import IDDDataModule
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger



def main():
    # torch.backends.cudnn.enabled = False
    # model = Net()
    # model = CustomFCN()
    # model = UNet_l(encChannels=(3, 16, 32, 64, 128, 256, 512), decChannels=(512,256,128,64,32,16), retainDim=False)
    # model = DeepnetLabV3_ResNet()
    model = EfficientFCN()
    
    # Datamodule with original IDD_lite train and validation data
    # idd_datamodule =  IDDDataModule('../idd20k_lite/leftImg8bit/train/*/*_image.jpg', '../idd20k_lite/leftImg8bit/val/*/*_image.jpg', 
    #                                 batch_size=16, num_workers=24)

    # Datamodule with training data of larger IDD and validation data of IDD_lite
    # idd_datamodule =  IDDDataModule('../full_dataset/small_idd20kII/leftImg8bit/train/*/*.jpg',
    #                                  '../idd20k_lite/leftImg8bit/val/*/*_image.jpg',  #Validation images have to be from idd_lite!
    #                                 batch_size=8, num_workers=24)

    # Datamodule with training data of larger IDD and smaller IDD_lite, and validation data of IDD_lite
    idd_datamodule =  IDDDataModule(#'../combined_dataset/leftImg8bit/train/*/*.jpg',
                                    '/data/divy/idd20k_lite/full_dataset/small_idd20kII/leftImg8bit/train/*/*.jpg',
                                    # '../combined_dataset/leftImg8bit/val/*/*.jpg',
                                     '../idd20k_lite/leftImg8bit/val/*/*_image.jpg',  #Validation images have to be from idd_lite!
                                    batch_size=32, num_workers=12)
    idd_datamodule.setup()
    train_dl = idd_datamodule.train_dataloader()
    val_dl = idd_datamodule.val_dataloader()

    checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="val_loss", mode='min', 
                                          filename="{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}")
    logger = pl.loggers.CSVLogger("runs/lightning_logs", name="IDD")
    # logger = TensorBoardLogger("runs/lightning_logs", name="IDD")

    trainer = pl.Trainer(strategy='ddp_find_unused_parameters_true', max_epochs=120, accelerator='gpu',
                          devices=[0], default_root_dir='runs', callbacks=[checkpoint_callback],
                          logger=logger, log_every_n_steps=54)
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)

if __name__ == '__main__':
    main()

