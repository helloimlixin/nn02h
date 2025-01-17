# A simple neural network for demo
from multiprocessing import freeze_support
from typing import Any

import torch
from torch import nn, optim, Tensor
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric


class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == target.shape
        self.correct += torch.sum(torch.eq(preds, target))
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()


class SimpleNet(pl.LightningModule):
    def __init__(self, in_channels, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(in_channels, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.my_accuracy = MyAccuracy()
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

        self.net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2
        )

    def forward(self, x):
        return self.net(x)

    def _inference_step(self, batch: Any, batch_index: int) -> tuple[Tensor, Any, Any]:
        """inference step for training, validation, and test steps
        :param batch: batched data input from dataloader
        :param batch_index: index of the batch in the dataloader
        :return: loss, scores, and labels
        """
        x, y = batch
        x = x.view(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)

        return loss, scores, y

    def training_step(self, batch, batch_index):
        loss, scores, y = self._inference_step(batch, batch_index)
        accuracy = self.my_accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict({"train_loss": loss, "train_accuracy": accuracy, "train_f1_score": f1_score},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_index):
        loss, scores, y = self._inference_step(batch, batch_index)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_index):
        loss, scores, y = self._inference_step(batch, batch_index)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_index):
        x, y = batch
        x = x.view(x.size(0), -1)
        scores = self.forward(x)
        predictions = torch.argmax(scores, dim=1)

        return predictions

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # single gpu, for downloading
        datasets.MNIST(root=self.data_dir, train=True, download=True)
        datasets.MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # multi gpus, load after prepare_data
        entire_dataset = datasets.MNIST(root=self.data_dir,
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=False)
        self.train_dataset, self.val_dataset = random_split(entire_dataset, [55000, 5000])
        self.test_dataset = datasets.MNIST(root=self.data_dir,
                                           train=False,
                                           transform=transforms.ToTensor(),
                                           download=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                            persistent_workers=True,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


if __name__ == "__main__":
    freeze_support()

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    in_features = 28 * 28
    n_classes = 10
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 5

    # Initialize network
    model = SimpleNet(in_channels=in_features, num_classes=n_classes).to(device)

    # Initialize data module
    datamodule = MNISTDataModule(data_dir="data", batch_size=batch_size, num_workers=4)

    trainer = pl.Trainer(accelerator="gpu", devices=[0], min_epochs=1, max_epochs=num_epochs, precision=64)
    # trainer.tune(model, train_loader)  # find the best hyperparameters
    torch.set_float32_matmul_precision('medium')
    trainer.fit(model, datamodule)
    trainer.validate(model, datamodule)
    trainer.test(model, datamodule)


