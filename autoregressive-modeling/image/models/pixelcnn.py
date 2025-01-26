import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as pl
from .utils import CausalConvolutionVStack, CausalConvolutionHStack, GatedMaskedCausalConvolution
from tqdm.auto import tqdm


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
        self.example_input_array = [torch.rand(3, in_channels, 32, 32), torch.randint(0, 10, (3,))]

    def forward(self, x, labels):
        x = (x.float() / 255.0)  # normalize the input to [0, 1]
        # initial vertical and horizontal stacks
        vstack = self.vconv(x)
        hstack = self.hconv(x)

        # gated masked causal convolutions
        for gated_conv in self.gated_convs:
            vstack, hstack = gated_conv(vstack, hstack, labels)

        # output layer
        out = self.conv_out(F.elu(hstack))

        # reshape the output to [Batch, Number of Classes, Number of Channels, Height, Width]
        out = out.reshape(out.size(0), 256, out.size(1) // 256, out.size(2), out.size(3))

        return out

    def compute_likelihood(self, x, labels):
        # compute the likelihood of the input
        preds = self.forward(x, labels)
        nll = F.cross_entropy(preds, x, reduction='none')
        bpd = nll.mean(dim=[1, 2, 3]) * np.log2(np.exp(1))

        return bpd.mean()

    @torch.no_grad()
    def generate(self, labels, img_size, img=None):
        """
        Sample from the autoregressive model.
        :param labels: labels for the generation
        :param condition: condition for the generation
        :param img_size: size of the image to generate
        :param img: initial image to start the generation if given
        :return: generated image
        """
        if img is None:
            img = torch.zeros(img_size).to(self.device)

        # generation
        for h in tqdm(range(img_size[2]), desc='Generating', leave=False):
            for w in range(img_size[3]):
                for c in range(img_size[1]):
                    # for efficient sampling, we only input the upper part of the image
                    preds = self.forward(img[:, :, :h + 1, :], labels)
                    probs = F.softmax(preds[:, :, c, h, w], dim=-1)
                    img[:, c, h, w] = torch.multinomial(probs, 1).squeeze(-1)

        return img / 255.0

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        # log input images
        if batch_idx == 0:
            self.logger.experiment.add_images('input_images', images[:8] / 255.0, self.current_epoch)
        loss = self.compute_likelihood(images, labels)
        self.log('train_bpd', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        loss = self.compute_likelihood(images, labels)
        self.log('val_bpd', loss)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        loss = self.compute_likelihood(images, labels)
        self.log('test_bpd', loss)
        return loss
