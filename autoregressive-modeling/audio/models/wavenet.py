from typing import Any

import lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from .utils import ResidualBlock, CausalDilatedConv1D, build_dilations, ResidualStack, DenseLayer


class WaveNet(pl.LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size, num_blocks, num_layers):
        super(WaveNet, self).__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._num_blocks = num_blocks
        self._num_layers = num_layers

        self.causal_conv1d = CausalDilatedConv1D(in_channels, in_channels, kernel_size, dilation=1)
        # residual_channels, skip_channels, kernel_size, num_blocks, num_layers
        self.residual_stack = ResidualStack(in_channels, out_channels, kernel_size, num_blocks, num_layers)

        self.dense = DenseLayer(out_channels, out_channels)

    def receptive_field(self):
        return sum([2 **  layer for layer in range(self._num_layers)] * self._num_blocks)

    def output_size(self, x):
        return int(x.size(2)) - self.receptive_field()

    def forward(self, x):
        x = self.causal_conv1d(x)
        skip = self.output_size(x)
        x, skip_outputs = self.residual_stack(x, skip)

        return self.dense(skip_outputs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        return loss

    def validation_step(self, *args: Any, **kwargs: Any):
        return self.training_step(*args, **kwargs)

    def test_step(self, *args: Any, **kwargs: Any):
        return self.training_step(*args, **kwargs)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)