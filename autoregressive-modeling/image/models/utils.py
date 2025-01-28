import torch
from torch import nn
from torch.nn import functional as F


class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        val, gate = x.chunk(2, dim=1)
        return self.tanh(val) * self.sigmoid(gate)


class GatedMaskedCausalConv(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual=True, n_classes=10):
        super().__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual

        self.class_cond_embedding = nn.Embedding(  # Embedding layer for class conditioning
            n_classes, 2 * dim
        )

        self.vstack = nn.Conv2d(
            dim, dim * 2,
            (kernel // 2 + 1, kernel), 1, (kernel // 2, kernel // 2)
        )

        self.v2h = nn.Conv2d(2 * dim, 2 * dim, 1)

        self.hstack = nn.Conv2d(
            dim, dim * 2,
            (1, kernel // 2 + 1), 1, (0, kernel // 2)
        )

        self.hres = nn.Conv2d(dim, dim, 1)

        self.gate = GatedActivation()

    def make_causal(self):
        """
        Mask the final row of the vertical stack and the final column of the horizontal stack. Only used when
        mask_type is 'A'.
        :return: None
        """
        self.vstack.weight.data[:, :, -1].zero_()  # mask final row of vertical stack
        self.hstack.weight.data[:, :, :, -1].zero_()  # mask final column of horizontal stack

    def forward(self, x_v, x_h, labels):
        if self.mask_type == 'A':
            self.make_causal()

        cond = self.class_cond_embedding(labels)
        h_vert = self.vstack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-1), :]
        out_v = self.gate(h_vert + cond[:, :, None, None])

        h_horiz = self.hstack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
        v2h = self.v2h(h_vert)

        out = self.gate(v2h + h_horiz + cond[:, :, None, None])
        if self.residual:
            out_h = self.hres(out) + x_h
        else:
            out_h = self.hres(out)

        return out_v, out_h


class GroupNorm(nn.Module):
    '''Group normalization module.

    References:
      - Wu, Y., & He, K. (2018). Group normalization. In Proceedings of the European Conference on Computer Vision
          (ECCV) (pp. 3-19).
    '''

    def __init__(self, num_channels, num_groups=32, eps=1e-6, affine=True):
        super(GroupNorm, self).__init__()

        self._num_channels = num_channels
        self._num_groups = num_groups
        self._eps = eps

        self._group_norm = nn.GroupNorm(num_groups=self._num_groups,
                                        num_channels=self._num_channels,
                                        eps=self._eps,
                                        affine=affine)

    def forward(self, x):
        return self._group_norm(x)


class Swish(nn.Module):
    '''Swish activation function.

    References:
      - Ramachandran, P., Zoph, B., & Le, Q. V. (2017). Swish: a self-gated activation function. arXiv preprint
          arXiv:1710.05941.
    '''

    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class ResNetBlock(nn.Module):
    '''Residual Block to build up the encoder and the decoder networks.

    References:
      - He, K., Zhang, X., Ren, S., & Sun, J. (2016).
          Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and
              pattern recognition (pp. 770-778).
    '''

    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(ResNetBlock, self).__init__()

        self.in_channels = in_channels
        self.num_hiddens = num_hiddens
        self.num_residual_hiddens = num_residual_hiddens

        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1,
                      stride=1,
                      bias=False)
        )

    def forward(self, x):
        return x + self.block(x) # skip connection


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            GroupNorm(out_channels),
            Swish(),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if self.in_channels != self.out_channels:  # handle dimension mismatch for skip connections
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)  # skip connection


class ResidualStack(nn.Module):
    '''Residual Stack for the encoder and decoder networks.

    References:
        - He, K., Zhang, X., Ren, S., & Sun, J. (2016).
            Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and
                pattern recognition (pp. 770-778).
    '''
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()

        self._num_residual_layers = num_residual_layers

        self._layers = nn.ModuleList([ResNetBlock(in_channels=in_channels, num_hiddens=num_hiddens,
                                                    num_residual_hiddens=num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)

        return F.relu(x)


class UpSampleBlock(nn.Module):
    '''Up-Sample Block, consists of a single convolutional layer and an additional
    prescale of the input features, resulting in a 2x up-sampling of the input.
    '''

    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')  # up-sample by nearest-neighbor interpolation
        return self.conv(x)


class DownSampleBlock(nn.Module):
    '''Just a reverse of UpSampleBlock, consists of a single convolutional layer and
    a padding operation before the downsampling operation, resulting in a 2x down-sampling
    of the input, off by 1 pixel.
    '''

    def __init__(self, channels) -> None:
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        padding = (0, 1, 0, 1)
        x = F.pad(x, padding, mode='constant', value=0)

        return self.conv(x)


class NonLocalBlockOriginal(nn.Module):
    '''Non-local Block to build up the encoder network based on the original paper.

    References:
        - Wang, X., Girshick, R., Gupta, A., & He, K. (2018). Non-local neural networks. In Proceedings of the IEEE
            conference on computer vision and pattern recognition (pp. 7794-7803).
    '''
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlockOriginal, self).__init__()

        self._sub_sample = sub_sample
        self._in_channels = in_channels
        self._inter_channels = inter_channels

        if self._inter_channels is None:
            self._inter_channels = in_channels // 2

        self._g = nn.Conv2d(in_channels=self._in_channels,
                            out_channels=self._inter_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0)

        if bn_layer:
            self._W = nn.Sequential(
                nn.Conv2d(in_channels=self._inter_channels,
                          out_channels=self._in_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0),
                GroupNorm(num_channels=self._in_channels)
            )
            nn.init.constant_(self._W[1].weight, 0) # initialize the weight of the GroupNorm layer to 0
            nn.init.constant_(self._W[1].bias, 0) # initialize the bias of the GroupNorm layer to 0
        else:
            self._W = nn.Conv2d(in_channels=self._inter_channels,
                                out_channels=self._in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0)
            nn.init.constant_(self._W.weight, 0)
            nn.init.constant_(self._W.bias, 0)

        if self._sub_sample:
            self._g = nn.Sequential(self._g, nn.MaxPool2d(kernel_size=(2, 2)))
            self._phi = nn.Sequential(self._phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self._g(x).view(batch_size, self._inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self._theta(x).view(batch_size, self._inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1) # transpose

        phi_x = self._phi(x).view(batch_size, self._inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self._inter_channels, *x.size()[2:])
        W_y = self._W(y)
        z = W_y + x # residual connection

        return z


class NonLocalBlock(nn.Module):
    '''
    Non-local block, used for long-range dependencies. It is a generalization of
    the self-attention mechanism. See,
    Wang, Xiaolong, et al. "Non-local neural networks."
    Proceedings of the IEEE conference on computer vision and
    pattern recognition. 2018.
    '''

    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = channels

        self.group_norm = GroupNorm(channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.projection = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        hidden = self.group_norm(x)
        q = self.q(hidden)
        k = self.k(hidden)
        v = self.v(hidden)

        # some reshaping for matrix multiplication
        b, c, h, w = q.shape

        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)

        attn = torch.bmm(q, k)  # batch matrix multiplication for attention
        attn = attn * (int(c) ** (-0.5))  # scaling factor
        attn = F.softmax(attn, dim=2)  # softmax along the last dimension to get the probabilies as attention weights
        attn = attn.permute(0, 2, 1)  # transpose for matrix multiplication

        hidden = torch.bmm(v, attn)
        hidden = hidden.reshape(b, c, h, w)

        hidden = self.projection(hidden)

        return x + hidden  # residual connection for baseline performance guarantee


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2,
                                 padding=1)

        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2,
                                 padding=1)

        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, x):
        x = self._conv_1(x)
        x = F.relu(x)
        x = self._conv_2(x)
        x = F.relu(x)
        x = self._conv_3(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                               out_channels=num_hiddens//2,
                                               kernel_size=4,
                                               stride=2,
                                               padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2,
                                               out_channels=3,
                                               kernel_size=4,
                                               stride=2,
                                               padding=1)

    def forward(self, x):
        x = self._conv_1(x)
        x = self._residual_stack(x)
        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)




