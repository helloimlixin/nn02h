import torch
from torch import nn, optim, Tensor
from typing import Any
import pytorch_lightning as pl
import torchmetrics
import torchvision

# from torchmetrics import Metric

# class MyAccuracy(Metric):
#     def __init__(self):
#         """
#         Demonstration of a custom accuracy metric.
#         """
#         super().__init__()
#         self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
#
#     def update(self, preds: Tensor, target: Tensor):
#         preds = torch.argmax(preds, dim=1)
#         assert preds.shape == target.shape
#         self.correct += torch.sum(torch.eq(preds, target))
#         self.total += target.numel()
#
#     def compute(self):
#         return self.correct.float() / self.total.float()


class SimpleNet(pl.LightningModule):
    def __init__(self, in_channels, learning_rate, num_classes):
        super(SimpleNet, self).__init__()
        self.lr = learning_rate
        self.fc1 = nn.Linear(in_channels, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

        self.net = nn.Sequential(self.fc1, nn.ReLU(), self.fc2)

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
        x, y = batch
        loss, scores, y = self._inference_step(batch, batch_index)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
                "train_f1_score": f1_score,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        if batch_index % 100 == 0:
            x = x[:8]
            grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
            self.logger.experiment.add_image("input_images", grid, self.global_step)

        return loss

    def validation_step(self, batch, batch_index):
        loss, scores, y = self._inference_step(batch, batch_index)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_index):
        loss, scores, y = self._inference_step(batch, batch_index)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict(
            {"test_loss": loss, "test_accuracy": accuracy, "test_f1_score": f1_score},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def predict_step(self, batch, batch_index):
        x, y = batch
        x = x.view(x.size(0), -1)
        scores = self.forward(x)
        predictions = torch.argmax(scores, dim=1)

        return predictions

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
