import lightning as pl
import torchmetrics
import torch

class IMDBSentimentClassifier(pl.LightningModule):
    def __init__(self, model, lr=2e-5):
        super().__init__()
        self.model = model
        self.lr = lr

        self.validation_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["label"])
        loss = outputs["loss"]
        self.log("train_loss", loss,
                 on_step=True, on_epoch=True, prog_bar=True)
        return loss  # passed to optimizer for backpropagation step (training)

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["label"])
        logits = outputs["logits"]
        predicted_labels = torch.argmax(logits, dim=1)
        self.validation_accuracy(predicted_labels, batch["label"])
        self.log("val_accuracy", self.validation_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["label"])
        logits = outputs["logits"]
        predicted_labels = torch.argmax(logits, dim=1)
        self.test_accuracy(predicted_labels, batch["label"])
        self.log("test_accuracy", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer