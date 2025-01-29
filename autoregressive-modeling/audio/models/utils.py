import torch
import torch.nn as nn
import numpy as np

def build_dilations(num_blocks, num_layers):
    dilations = []

    for block in range(num_blocks):
        block_dilations = []
        for layer in range(num_layers):
            block_dilations.append(2 ** block)
        dilations.append(block_dilations)

    return dilations

class CausalDilatedConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super(CausalDilatedConv1D, self).__init__()
        self.block_out = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias, padding='same')

    def forward(self, x):
        return self.conv(x)[:, :, :-self.block_out]

class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, skip_channels, kernel_size, dilation, bias=True):
        super(ResidualBlock, self).__init__()
        self.causal_dilated_conv1d = CausalDilatedConv1D(residual_channels, residual_channels, kernel_size, dilation)
        self.residual_conv1d = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip_conv1d = nn.Conv1d(residual_channels, skip_channels, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, skip):
        x = self.causal_dilated_conv1d(inputs)
        x = self.tanh(x) * self.sigmoid(x)

        residual_output = self.residual_conv1d(x)
        residual_output = residual_output + inputs[..., -residual_output.size(2):]
        skip_output = self.skip_conv1d(x)
        skip_output = skip_output[..., -skip:]

        return residual_output, skip_output

class ResidualStack(nn.Module):
    def __init__(self, residual_channels, skip_channels, kernel_size, num_blocks, num_layers):
        super(ResidualStack, self).__init__()

        self._residual_channels = residual_channels
        self._skip_channels = skip_channels
        self._kernel_size = kernel_size

        dilations = build_dilations(num_blocks, num_layers)

        self.blocks = nn.ModuleList()

        for dilation_layer in dilations:
            for dilation in dilation_layer:
                self.blocks.append(ResidualBlock(residual_channels, skip_channels, kernel_size, dilation))

    def forward(self, x, skip):
        residual_output = x
        skip_outputs = []
        for residual_block in self.blocks:
            residual_output, skip_output = residual_block(residual_output, skip)
            skip_outputs.append(skip_output)
        return residual_output, torch.stack(skip_outputs)


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.relu = nn.ReLU()
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, skip_connections):
        # batch, channels, time -> input data shape
        output = torch.sum(skip_connections, dim=0)  # sum over time

        for _ in range(2):
            output = self.relu(output)
            output = self.conv1d(output)

        return self.softmax(output)

