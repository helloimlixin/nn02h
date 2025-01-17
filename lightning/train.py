from multiprocessing import freeze_support

import torch
from dataset import MNISTDataModule
from model import SimpleNet
import config
import pytorch_lightning as pl
from callbacks import MyPrintingCallback, EarlyStopping

if __name__ == "__main__":
    freeze_support()

    # Initialize network
    model = SimpleNet(in_channels=config.INPUT_SIZE,
                      learning_rate=config.LEARNING_RATE,
                      num_classes=config.NUM_CLASSES)

    # Initialize data module
    datamodule = MNISTDataModule(data_dir=config.DATA_DIR,
                                 batch_size=config.BATCH_SIZE,
                                 num_workers=config.NUM_WORKERS)

    trainer = pl.Trainer(accelerator=config.ACCELERATOR,
                         devices=config.DEVICES,
                         min_epochs=1, max_epochs=config.NUM_EPOCHS,
                         precision=config.PRECISION,
                         callbacks=[MyPrintingCallback(), EarlyStopping(monitor="val_loss")])
    # trainer.tune(model, train_loader)  # find the best hyperparameters
    torch.set_float32_matmul_precision('medium')
    trainer.fit(model, datamodule)
    trainer.validate(model, datamodule)
    trainer.test(model, datamodule)
