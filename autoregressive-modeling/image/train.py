import os
from multiprocessing import freeze_support
from datamodules.mnist import MNISTDataModule
from models.pixelcnn import PixelCNN
import config
import torch
import lightning as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger
# from lightning.pytorch.strategies import DeepSpeedStrategy
import warnings
warnings.filterwarnings('ignore')

pl.seed_everything(42)

# set deterministic for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

progress_bar = RichProgressBar(  ## wip
    theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="green1",
        batch_progress="green_yellow",
        time="grey82",
        processing_speed="grey82",
        metrics="grey82",
    ))

if __name__ == "__main__":
    freeze_support()

    logger = TensorBoardLogger("tb_logs", name="pixelcnn_v1")
    # strategy = DeepSpeedStrategy()

    # init datamodule
    dm = MNISTDataModule(config.DATA_DIR,
                         batch_size=config.BATCH_SIZE,
                         num_workers=config.NUM_WORKERS)

    # init model
    model = PixelCNN(config.NUM_CHANNELS, config.NUM_HIDDENS)

    # init trainer
    trainer = pl.Trainer(default_root_dir=os.path.join(config.CHECKPOINT_DIR, 'pixelcnn'),
                         logger=logger,
                         # strategy='ddp',
                         accelerator='gpu' if str(device) == 'cuda' else 'cpu',
                         devices=1, max_epochs=10,
                         check_val_every_n_epoch=1,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode='min', monitor='val_bpd'),
                                    LearningRateMonitor(logging_interval='epoch'),
                                    progress_bar])

    torch.set_float32_matmul_precision("medium")
    trainer.fit(model, dm)


