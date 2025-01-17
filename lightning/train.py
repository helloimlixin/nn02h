from multiprocessing import freeze_support
import torch
from dataset import MNISTDataModule
from model import SimpleNet
import config
import lightning as pl
from callbacks import MyPrintingCallback, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.strategies import DeepSpeedStrategy

if __name__ == "__main__":
    freeze_support()

    logger = TensorBoardLogger("tb_logs", name="mnist_model_v1")
    strategy = DeepSpeedStrategy()
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
        scheduler=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20)
    )

    # Initialize network
    model = SimpleNet(
        in_channels=config.INPUT_SIZE,
        learning_rate=config.LEARNING_RATE,
        num_classes=config.NUM_CLASSES,
    )

    # Initialize data module
    datamodule = MNISTDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    trainer = pl.Trainer(
        strategy=strategy,  # copy the model to each GPU, linear scaling
        profiler=profiler,
        logger=logger,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION,
        callbacks=[MyPrintingCallback(), EarlyStopping(monitor="val_loss")],
    )
    # trainer.tune(model, train_loader)  # find the best hyperparameters
    torch.set_float32_matmul_precision("medium")
    trainer.fit(model, datamodule)
