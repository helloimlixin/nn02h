import torch
from torch import nn


class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        val, gate = x.chunk(2, dim=1)
        return torch.tanh(val) * torch.sigmoid(gate)


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


class VectorQuantizer(nn.Module):
    pass


