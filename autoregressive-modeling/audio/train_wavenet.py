import os
import pandas as pd
import lightning as pl
import torch
from datamodules.liverpool_ion_switching import LiverpoolIonSwitchingDataModule
from datamodules.vctk import VCTKDataModule, VCTKDataset, get_dataset
from torch.utils.data import DataLoader
from models.wavenet import WaveNet
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

train_df = pd.read_csv('../../data/liverpool-ion-switching/train.csv')
test_df = pd.read_csv('../../data/liverpool-ion-switching/test.csv')

dataset_name = 'vctk'

logger = TensorBoardLogger("tensorboard_logs", name=f"wavenet_{dataset_name}")
# strategy = DeepSpeedStrategy()

# init datamodule
dm = None
if dataset_name == 'vctk':
    dm = VCTKDataModule('../../data/VCTK/VCTK-Corpus/VCTK-Corpus', batch_size=4, num_workers=0)
elif dataset_name == 'liverpool-ion-switching':
    dm = LiverpoolIonSwitchingDataModule(train_df, test_df,
                                         batch_size=256,
                                         sequence_length=1000,
                                         num_workers=0)

# init model
model = WaveNet(in_channels=256, residual_channels=256, skip_channels=32, num_blocks=5)

# init progress bar
progress_bar = RichProgressBar(
    theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="green1",
        batch_progress="green_yellow",
        time="grey82",
        processing_speed="grey82",
        metrics="grey82",
    ))

# init callbacks
lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint = ModelCheckpoint(monitor='val_loss')

# init trainer
trainer = pl.Trainer(logger=logger,
                     # strategy='ddp',
                     accelerator='gpu' if str(device) == 'cuda' else 'cpu',
                     devices=1, max_epochs=10,
                     check_val_every_n_epoch=1,
                     callbacks=[ModelCheckpoint(mode='min', monitor='val_loss',
                                                filename='vae-{epoch:02d}-{val_loss:.2f}',
                                                # dirpath=os.path.join(config.CHECKPOINT_DIR, 'vae'),
                                                verbose=False, save_last=True, save_top_k=5,
                                                save_on_train_epoch_end=True),
                                LearningRateMonitor(logging_interval='epoch'),
                                progress_bar])

torch.set_float32_matmul_precision("medium")
# if os.path.exists(os.path.join(config.CHECKPOINT_DIR, 'vae', 'last.ckpt')):
#     trainer.fit(model, ckpt_path=os.path.join(config.CHECKPOINT_DIR, 'vae', 'last.ckpt'), datamodule=dm)
# else:
#     trainer.fit(model, datamodule=dm)

trainer.fit(model, datamodule=dm)
# test
test_results = trainer.test(model, dm)


