import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as pl
from .utils import Encoder, Decoder
from .laser import VectorQuantizer, VectorQuantizerEMA
from torchvision.utils import make_grid
import numpy as np


class VQVAE(pl.LightningModule):
    def __init__(self, in_channels=3, num_hiddens=256, num_residual_hiddens=32, quantizer = "Vanilla",
                 num_residual_layers=2, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.save_hyperparameters()

        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim

        self.encoder = Encoder(in_channels, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(num_hiddens, embedding_dim, 1, stride=1)
        if quantizer == "Vanilla":
            self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        elif quantizer == "EMA":
            self.quantizer = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

    def encode(self, x):
        """ Encodes an image into quantized latent representation indices. """
        z = self.encoder(x)
        z = self._pre_vq_conv(z)
        encoding_indices = self.quantizer(z)
        return encoding_indices, z

    def decode(self, encoding_indices):
        """ Decodes quantized latent representation indices back into an image. """
        quantized, encodings = self.quantizer.quantize(encoding_indices)
        quantized = quantized.view(-1, 8, 8, self._embedding_dim)
        # x_recon = self.decoder(quantized.permute(0, 3, 1, 2).contiguous())

        return quantized, encodings

    def forward(self, x):
        """ Full forward pass of VQVAE (encode → quantize → decode). """
        encoding_indices, z = self.encode(x)
        quantized, encodings = self.decode(encoding_indices)
        loss, quantized, perplexity, encodings = self.quantizer.loss(quantized, encodings, z)
        x_recon = self.decoder(quantized)

        return loss, x_recon, perplexity, encoding_indices

    def training_step(self, batch, batch_idx):
        x, _ = batch
        laser_loss, reconstruction, _, _ = self.forward(x)
        loss = laser_loss + F.mse_loss(reconstruction, x)
        self.log('train_loss', loss,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        laser_loss, reconstruction, _, _ = self.forward(x)
        loss = laser_loss + F.mse_loss(reconstruction, x)
        x = (x + 1) / 2
        reconstruction = (reconstruction + 1) / 2
        self.log('val_loss', loss)
        self.logger.experiment.add_image("input_images", make_grid(x[:16], nrow=4), self.global_step)
        self.logger.experiment.add_image("reconstructed_images", make_grid(reconstruction[:16], nrow=4), self.global_step)
        return loss

    def test_step(self, batch, batch_idx):
        x, _ = batch
        laser_loss, reconstruction, _, _ = self.forward(x)
        loss = laser_loss + F.mse_loss(reconstruction, x)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=2e-4)

    def generate(self, x):
        encodings = self.encode(x)
        decoded = self.decode(encodings)
        return decoded



