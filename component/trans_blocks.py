from torch import nn
import torch
from utils import autopad


class Focus(nn.Module):
    """Focus部分主要是把图像拼在一起增加通道数"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()

        self.conv1 = nn.Conv2d(c1 * 4, c2, k, s, padding=autopad(k, p), groups=g)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(True)

    def forward(self, x):
        cat = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1)
        return self.act(self.bn(self.conv1(cat)))
