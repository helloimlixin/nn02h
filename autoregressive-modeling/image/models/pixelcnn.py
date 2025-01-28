import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as pl
from .utils import GatedActivation, GatedMaskedCausalConv
from tqdm.auto import tqdm


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            pass


class PixelCNN(pl.LightningModule):
    def __init__(self, in_channels=512, num_hiddens=64, num_layers=15, num_classes=10):
        super().__init__()
        self.save_hyperparameters()  # save the hyperparameters to the checkpoint

        self._in_channels = in_channels
        self.embedding = nn.Embedding(in_channels, num_hiddens)

        self.layers = nn.ModuleList()

        # stack of gated masked causal convolutions with dilation
        for i in range(num_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel_size = 7 if i == 0 else 3
            residual = False if i == 0 else True

            self.layers.append(
                GatedMaskedCausalConv(mask_type, num_hiddens, kernel_size, residual, num_classes)
            )

        # final 1x1 convolution to map to the output channels
        self.conv_out = nn.Sequential(
            nn.Conv2d(num_hiddens, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, in_channels, 1)
        )

        self.ae = None

        self.apply(init_weights)

    def forward(self, x, labels):
        x_shape = x.size() + (-1,)
        x = self.embedding(x.view(-1)).view(x_shape)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x_v, x_h = x, x

        for layer in self.layers:
            x_v, x_h = layer(x_v, x_h, labels)

        return self.conv_out(x_h)

    def compute_likelihood(self, x, labels):
        target = x
        # compute the likelihood of the input
        logits = self.forward(x, labels)
        logits = logits.permute(0, 2, 3, 1).contiguous()
        nll = F.cross_entropy(logits.view(-1, self._in_channels), target.view(-1), reduction='none').view_as(target)

        bpd = nll.mean(dim=[1, 2]) / np.log(2)

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
            img = torch.zeros(img_size, dtype=torch.long).to(self.device)

        # generation
        for h in tqdm(range(img_size[1]), desc='Generating', leave=False):
            for w in range(img_size[2]):
                logits = self.forward(img, labels)
                probs = F.softmax(logits[:, :, h, w], dim=-1)
                img[:, h, w] = torch.multinomial(probs, 1).squeeze(-1)

        return img / 255.0

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        # log input images
        if batch_idx == 0:
            self.logger.experiment.add_images('input_images', images[:8], self.current_epoch)
        if self.ae is not None:
            self.ae.eval()
            latents = self.ae.encode(images).detach().long()
            logits = self.forward(latents, labels)
            logits = logits.permute(0, 2, 3, 1).contiguous()

            loss = F.cross_entropy(logits.view(-1, self._in_channels), latents.view(-1))
        else:
            images = (images[:, 0] * 255).long()
            loss = self.compute_likelihood(images, labels)
        self.log('train_bpd', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        # images = (images[:, 0] * 255).long()
        if self.ae is not None:
            latents = self.ae.encode(images).detach().long()
            logits = self.forward(latents, labels)
            logits = logits.permute(0, 2, 3, 1).contiguous()

            loss = F.cross_entropy(logits.view(-1, self._in_channels), latents.view(-1))
        else:
            images = (images[:, 0] * 255).long()
            loss = self.compute_likelihood(images, labels)
        self.log('val_bpd', loss)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        # images = (images[:, 0] * 255).long()
        if self.ae is not None:
            latents = self.ae.encode(images).detach().long()
            logits = self.forward(latents, labels)
            logits = logits.permute(0, 2, 3, 1).contiguous()

            loss = F.cross_entropy(logits.view(-1, self._in_channels), latents.view(-1))
        else:
            images = (images[:, 0] * 255).long()
            loss = self.compute_likelihood(images, labels)
        self.log('test_bpd', loss)
        return loss
