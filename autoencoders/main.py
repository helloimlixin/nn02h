import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torchvision.datasets import MNIST  # using MNIST for simple demonstration
from sklearn.decomposition import PCA

# define data transformations
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)), # for MNIST dataset, we resize the image to 32x32 for convenience purpose
        transforms.ToTensor(), # convert the image to a tensor
    ]
)

# load the MNIST dataset
train_dataset = MNIST(root='../../data', train=True, download=True, transform=transform)
test_dataset = MNIST(root='../../data', train=False, download=True, transform=transform)

# define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



# create data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)