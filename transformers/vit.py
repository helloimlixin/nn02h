import torch # this implementation is based on PyTorch
import torch.nn as nn
import torchvision.transforms as T # input image manipulations (resizing, converting to tensors, etc.
from torch.optim import Adam # using the Adam optimizer
from torchvision.datasets.mnist import MNIST # using MNIST dataset for minimal experiments
from torch.utils.data import DataLoader # for dataloader
import numpy as np # basic numerical operations like sine and cosine for positional embeddings

class PatchEmbedding(nn.Module):
    '''
    The Patch Embedding layer.
    '''
    def __init__(self, feature_dim, img_dim, patch_dim, num_channels):
        super().__init__()

        self.feature_dim = feature_dim # latent feature dimension
        self.img_dim = img_dim # input image dimension
        self.patch_dim = patch_dim # patch dimension
        self.num_channels = num_channels # number of channels

        self.linear_project = nn.Conv2d(
                in_channels=self.num_channels,
                out_channels=self.feature_dim,
                kernel_size=patch_dim,
                stride=patch_dim)

    def forward(self, x):
        x = self.linear_project(x) # (B, C, H, W) -> (B, feature_dim, patch_height, patch_width)
        x = x.flatten(2) # (B, feature_dim, patch_height, patch_width) -> (B, feature_dim, patch_height x patch_width)
        x = x.transpose(1, 2) # (B, feature_dim, patch_height x patch_dim) -> (B, patch_height x patch_width, feature_dim)

        return x





