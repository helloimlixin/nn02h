import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Conv1d):
    """Causal 1D convolution with proper padding adjustment."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        padding = (kernel_size - 1) * dilation  # Calculate appropriate padding
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size,
                                           padding=padding, dilation=dilation)

    def forward(self, x):
        out = super(CausalConv1d, self).forward(x)

        # Check if padding slice is necessary
        if self.padding[0] > 0:
            return out[:, :, :-self.padding[0]]  # Maintain causality by slicing correctly
        return out



class ResidualBlock(nn.Module):
    """Residual block with a gated activation unit."""
    def __init__(self, residual_channels, skip_channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.causal_conv = CausalConv1d(residual_channels, residual_channels, kernel_size, dilation)

        # Gate mechanism (tanh and sigmoid)
        self.conv_tanh = CausalConv1d(residual_channels, residual_channels, kernel_size, dilation)
        self.conv_sigmoid = CausalConv1d(residual_channels, residual_channels, kernel_size, dilation)

        # 1x1 convolutions for residual and skip connections
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, 1)

    def forward(self, x):
        # Causal convolution followed by gated activation
        tanh_out = torch.tanh(self.conv_tanh(x))
        sigmoid_out = torch.sigmoid(self.conv_sigmoid(x))
        gated_out = tanh_out * sigmoid_out

        # Residual and skip connections
        residual_out = self.residual_conv(gated_out) + x  # Residual connection
        skip_out = self.skip_conv(gated_out)  # Skip connection

        return residual_out, skip_out


class WaveNetModule(nn.Module):
    """Simplified WaveNet model for audio generation."""
    def __init__(self, in_channels, residual_channels, skip_channels, num_blocks, kernel_size=2):
        super(WaveNetModule, self).__init__()
        self.initial_causal_conv = CausalConv1d(in_channels, residual_channels, kernel_size=1)

        # Create a series of residual blocks with exponentially increasing dilations
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(residual_channels, skip_channels, kernel_size, dilation=2**i)
            for i in range(num_blocks)
        ])

        # Final layers for output generation
        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(skip_channels, in_channels, kernel_size=1)
        )

    def forward(self, x):
        # Initial causal convolution
        x = self.initial_causal_conv(x)

        skip_connections = []
        for block in self.residual_blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        # Sum all skip connections and pass through the final layers
        skip_sum = torch.sum(torch.stack(skip_connections), dim=0)
        return self.output_layer(skip_sum)

# Example usage:
if __name__ == "__main__":
    # Random input of shape (batch_size, in_channels, seq_length)
    x = torch.randn(1, 1, 100)  # For example, a 1D signal of length 100
    model = WaveNetModule(in_channels=1, residual_channels=32, skip_channels=32, num_blocks=6)
    output = model(x)
    print("Output shape:", output.shape)
