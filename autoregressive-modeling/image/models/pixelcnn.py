import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as pl
from .utils import CausalConvolutionVStack, CausalConvolutionHStack, GatedMaskedCausalConvolution

class PixelCNN(pl.LightningModule):
    def __init__(self, in_channels, num_hiddens):
        super().__init__()
        self.save_hyperparameters()  # save the hyperparameters to the checkpoint

        self.vconv = CausalConvolutionVStack(in_channels, num_hiddens, mask_center=True)
        self.hconv = CausalConvolutionHStack(in_channels, num_hiddens, mask_center=True)

        # stack of gated masked causal convolutions with dilation
        self.gated_convs = nn.ModuleList(
            [
                GatedMaskedCausalConvolution(num_hiddens),
                GatedMaskedCausalConvolution(num_hiddens, dilation=2),
                GatedMaskedCausalConvolution(num_hiddens),
                GatedMaskedCausalConvolution(num_hiddens, dilation=4),
                GatedMaskedCausalConvolution(num_hiddens),
                GatedMaskedCausalConvolution(num_hiddens, dilation=2),
                GatedMaskedCausalConvolution(num_hiddens)
            ]
        )

        # final 1x1 convolution to map to the output channels
        self.conv_out = nn.Conv2d(num_hiddens, in_channels * 256, 1, padding=0)
        self.example_input_array = torch.rand(1, in_channels, 28, 28)  # for MNIST

    def forward(self, x):
        # scale the input to the range [-1, 1]
        x = (x.float() / 255.0) * 2 - 1

        # initial vertical and horizontal stacks
        vstack = self.vconv(x)
        hstack = self.hconv(x)

        # gated masked causal convolutions
        for gated_conv in self.gated_convs:
            vstack, hstack = gated_conv(vstack, hstack)

        # output layer
        out = self.conv_out(F.elu(hstack))

        # reshape the output to [Batch, Number of Classes, Number of Channels, Height, Width]
        out = out.reshape(out.size(0), 256, out.size(1) // 256, out.size(2), out.size(3))

        return out

    def compute_likelihood(self, x):
        # compute the likelihood of the input
        preds = self.forward(x)
        nll = F.cross_entropy(preds, x, reduction='none')
        bpd = nll.mean(dim=[1, 2, 3]) * np.log2(np.exp(1))

        return bpd.mean()

    @torch.no_grad()
    def sample(self, img_size, img=None):
        """
        Sample from the autoregressive model.
        :param img_size: size of the image to generate
        :param img: initial image to start the generation if given
        :return: generated image
        """
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        loss = self.compute_likelihood(batch[0])
        self.log('train_bpd', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_likelihood(batch[0])
        self.log('val_bpd', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_likelihood(batch[0])
        self.log('test_bpd', loss)
        return loss