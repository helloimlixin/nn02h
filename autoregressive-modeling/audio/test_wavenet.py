from unittest import TestCase
from models.wavenet import WaveNet
import torch
from torch import nn
from einops import repeat


class TestWaveNet(TestCase):

    def setUp(self):
        super().setUp()
        in_channels = 1
        out_channels = 1
        self.sample_size = 2000
        self.batch_size = 8
        self.model = WaveNet(in_channels, out_channels, 2, 2, 5)
        self.input = nn.Parameter(torch.randn(in_channels, self.sample_size))
        self.input = repeat(self.input, 'c t -> b c t', b=self.batch_size)  # einops repeat to add batch dimension

    def test_run(self):
        y = self.model(self.input)
        print(y.size())  # torch.Size([8, 1, 1938])

    def test_receptive_field(self):
        print(self.model.calculateReceptiveField())  # 62
        print(self.model.calculateOutputSize(self.input))  # 1938


