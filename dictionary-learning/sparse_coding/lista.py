import torch
import torch.nn as nn
import torch.nn.functional as F

class LISTA(nn.Module):
    def __init__(self, signal_size, dictionary_size, num_layers):
        """
        LISTA module for sparse coding.

        Args:
            signal_size (int): Dimensionality of the input signals (e.g., 192 for 8x8 RGB patches).
            dictionary_size (int): Number of dictionary atoms (e.g., 512).
            num_layers (int): Number of LISTA layers (fixed depth).
        """
        super(LISTA, self).__init__()
        self.num_layers = num_layers

        # Trainable parameters
        self.W = nn.Parameter(torch.randn(signal_size, dictionary_size))  # Encoder matrix
        self.S = nn.Parameter(torch.eye(dictionary_size) - 0.1 * torch.mm(
            torch.randn(signal_size, dictionary_size).T, torch.randn(signal_size, dictionary_size)
        ))  # Feedback matrix
        self.threshold = nn.Parameter(torch.ones(1) * 0.05)  # Soft-thresholding parameter

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights to ensure stability at the start."""
        with torch.no_grad():
            self.W.data = F.normalize(self.W.data, dim=0)

    def forward(self, x):
        """
        Perform sparse coding using LISTA.

        Args:
            x (torch.Tensor): Input signals of shape (batch_size, signal_size).

        Returns:
            torch.Tensor: Sparse codes of shape (batch_size, dictionary_size).
        """
        # Initialize sparse codes as zero
        batch_size = x.size(0)
        z = torch.zeros(batch_size, self.W.size(1), device=x.device)  # Shape: (batch_size, dictionary_size)

        for _ in range(self.num_layers):
            # LISTA update step
            z = z + torch.matmul(x, self.W) - torch.matmul(z, self.S.T)  # Gradient descent approximation
            z = F.softshrink(z, lambd=self.threshold.item())  # Apply soft-thresholding for sparsity

        return z
