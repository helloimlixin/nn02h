import torch
from torch import nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha):
        """
        lora layer implementation, which is usually applied to a neural network's linear (feedforward) layers.
        :param in_features: input dimension of the layer we are going to apply LoRA
        :param out_features: the respective output dimension of the layer we are going to apply LoRA
        :param rank: a hyperparameter that determines the rank of the low-rank matrices used for adaptation
        :param alpha: a hyperparameter that determines the magnitude of the weight changes introduced LoRA layer
        """
        super().__init__()

        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self._A = nn.Parameter(torch.randn(in_features, rank) * std_dev)
        self._B = nn.Parameter(torch.zeros(rank, out_features))
        self._alpha = alpha

    def forward(self, x):
        x = self._alpha * (x @ self._A @ self._B)
        return x

class LoRALinear(nn.Module):
    def __init__(self, linear, rank, alpha):
        """
        lora linear layer implementation, which is usually applied to a neural network's linear (feedforward) layers.
        :param linear: the linear layer we are going to apply LoRA
        :param rank: a hyperparameter that determines the rank of the low-rank matrices used for adaptation
        :param alpha: a hyperparameter that determines the magnitude of the weight changes introduced LoRA layer
        """
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)
