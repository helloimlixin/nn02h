import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    """A simple autoencoder model. The encoder network is a stack of linear layers with ReLU activation functions
    as nonlinearities. The autoencoder network is trained to learn a compressed representation of the data.
    """
    def __init__(self, bottleneck_dim: int=32):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, bottleneck_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32 * 32),
            nn.Sigmoid() # ensure the output is in the range [0, 1] for visualization and reconstruction
                         # to ensure the output is in the range [-1, 1], we can use nn.Tanh() instead of nn.Sigmoid()
        )