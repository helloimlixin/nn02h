from transformers import AutoModelForSequenceClassification
from functools import partial
from lora import LoRALinear
from utils import count_trainable_params, IMDBDataModule
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import CSVLogger
import torch
from imdb_sentiment_classifier import IMDBSentimentClassifier
import warnings
warnings.filterwarnings("ignore")


model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)

for param in model.parameters():
    param.requires_grad = False

layers = []

# define hyperparameters for LoRA
lora_rank = 8
lora_alpha = 16
lora_dropout = 0.05
lora_query = True
lora_key = False
lora_value = True
lora_projection = False
lora_mlp = False
lora_head = False

lora_assign = partial(LoRALinear, rank=lora_rank, alpha=lora_alpha)

for layer in model.distilbert.transformer.layer:
    if lora_query:
        layer.attention.q_lin = lora_assign(layer.attention.q_lin)
    if lora_key:
        layer.attention.k_lin = lora_assign(layer.attention.k_lin)
    if lora_value:
        layer.attention.v_lin = lora_assign(layer.attention.v_lin)
    if lora_projection:
        layer.attention.out_lin = lora_assign(layer.attention.out_lin)
    if lora_mlp:
        layer.ffn.lin1 = lora_assign(layer.ffn.lin1)
        layer.ffn.lin2 = lora_assign(layer.ffn.lin2)

if lora_head:
    model.pre_classifier = lora_assign(model.pre_classifier)
    model.classifier = lora_assign(model.classifier)

# count number of parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")

print(f"Number of trainable parameters: {count_trainable_params(model)}")

imdb_sentiment_classifier = IMDBSentimentClassifier(model)

imdb_datamodule = IMDBDataModule()

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

callbacks = [
    ModelCheckpoint(
        monitor="val_accuracy",
        filename="imdb-sentiment-{epoch:02d}-{val_accuracy:.2f}",
        save_top_k=1,
        mode="max",
    ), progress_bar]

logger = CSVLogger("logs", name="imdb-sentiment")

# define the trainer
trainer = pl.Trainer(
    max_epochs=10,
    callbacks=callbacks,
    accelerator="gpu",
    devices=1,
    logger=logger,
    log_every_n_steps=10
)

torch.set_float32_matmul_precision('medium')
trainer.fit(imdb_sentiment_classifier, imdb_datamodule)