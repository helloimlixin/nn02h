import torch
from torch import nn

class MaskedCausalConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, mask, dilation=1):
        """
        Convolution kernel with mask applied to the weights to ensure causality.
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param mask: tensor of shape [kernel_size_H, kernel_size_W] with 0s for
                        weights that should be masked and 1s for weights that should
                        be kept.
        """
        super().__init__()
        kernel_size = (mask.size(0), mask.size(1))
        padding = tuple([dilation * (kernel_size[0] - 1) // 2,
                         dilation * (kernel_size[1] - 1) // 2])

        # define the convolution layer
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding
        )
        # mask as buffer to ensure it is moved to the same device as the model
        # and not treated as a model parameter
        self.register_buffer("mask", mask[None, None])

    def forward(self, x):
        self.conv.weight.data *= self.mask  # apply the mask
        return self.conv(x)


class CausalConvolutionVStack(MaskedCausalConvolution):
    def __init__(self, in_channels, out_channels, kernel_size=3, mask_center=False):
        """
        Vertical stack of causal convolution layers. The mask is applied to the weights to ensure causality,
        which is applied to mask out all pixels below the current pixel.
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: size of the convolution kernel
        :param mask_center: for the first convolutional layer, mask the center pixel
        """
        mask = torch.ones(kernel_size, kernel_size)
        mask[kernel_size // 2 + 1:, :] = 0

        # for the first layer, mask the center pixel
        if mask_center:
            mask[kernel_size // 2, :] = 0
        super().__init__(in_channels, out_channels, mask)


class CausalConvolutionHStack(MaskedCausalConvolution):
    def __init__(self, in_channels, out_channels, kernel_size=3, mask_center=False):
        """
        Horizontal stack of causal convolution layers. The mask is applied to the weights to ensure causality,
        which is applied to mask out all pixels to the left of the current pixel.
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: size of the convolution kernel, the kernel has a size of 1 in the vertical direction
            because we only care about the horizontal direction
        :param mask_center: for the first convolutional layer, mask the center pixel
        """
        mask = torch.ones(1, kernel_size)
        mask[0, kernel_size // 2 + 1 :] = 0

        # for the first layer, mask the center pixel
        if mask_center:
            mask[0, kernel_size // 2] = 0

        super().__init__(in_channels, out_channels, mask)


class GatedMaskedCausalConvolution(nn.Module):
    def __init__(self, in_channels, dilation=1):
        super().__init__()
        self.vconv = CausalConvolutionVStack(in_channels, 2 * in_channels)
        self.hconv = CausalConvolutionHStack(in_channels, 2 * in_channels)
        self.v2h = nn.Conv2d(2 * in_channels, 2 * in_channels, 1, padding=0)
        self.hconv_1x1 = nn.Conv2d(in_channels, in_channels, 1, padding=0)

    def forward(self, vstack, hstack):
        # vertical stack computation on the left
        vstack_features = self.vconv(vstack)
        vstack_val, vstack_gate = vstack_features.chunk(2, dim=1)
        vstack_out = torch.tanh(vstack_val) * torch.sigmoid(vstack_gate)  # element-wise multiplication

        # horizontal stack computation on the right
        hstack_features = self.hconv(hstack)
        # use horizontal stack as output
        hstack_features = hstack_features + self.v2h(vstack_features)
        hstack_val, hstack_gate = hstack_features.chunk(2, dim=1)
        hstack_features = torch.tanh(hstack_val) * torch.sigmoid(hstack_gate)
        hstack_out = self.hconv_1x1(hstack_features)
        hstack_out = hstack_out + hstack  # residual connection

        return vstack_out, hstack_out


