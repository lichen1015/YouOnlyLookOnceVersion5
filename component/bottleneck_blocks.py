from torch import nn
from utils import autopad
import torch


class SimBottleneck(nn.Module):
    """
        bottleneck简单实现
    """

    def __init__(self, cin, cout, expansion=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(cin * expansion)

        self.conv1 = nn.Conv2d(cin, c_, 1, 1, padding=autopad(1))
        self.bn = nn.BatchNorm2d(c_)
        self.act = nn.Hardswish(True)

        self.conv2 = nn.Conv2d(c_, cout, 3, 1, groups=g, padding=autopad(3))
        self.bn2 = nn.BatchNorm2d(cout)

        self.add = shortcut and cin == cout

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        residual = self.act(self.bn2(self.conv2(out)))
        return x + residual if self.add else residual


class SimBottleneckCSP(nn.Module):

    def __init__(self, cin, cout, n=1, expansion=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(cin * expansion)

        # convBase
        self.conv1 = nn.Conv2d(cin, c_, 1, 1, padding=autopad(1),groups=g)
        self.bn1 = nn.BatchNorm2d(c_)
        self.act1 = nn.Hardswish(True)

        # simBottleneck
        self.m = nn.Sequential(*[SimBottleneck(c_, c_, expansion=1, g=g, shortcut=shortcut) for _ in range(n)])

        # conv2
        self.conv2 = nn.Conv2d(cin, c_, 1, 1, bias=False)
        # conv3
        self.conv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)

        # convBase2
        self.conv4 = nn.Conv2d(2 * c_, cout, 1, 1, padding=autopad(1),groups=g)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act2 = nn.LeakyReLU(True)

    def forward(self, x):
        out_part1 = self.act1(self.bn1(self.conv1(x)))
        y1 = self.conv3(self.m(out_part1))

        y2 = self.conv2(x)

        out = self.conv4(self.act2(self.bn2(torch.cat([y1, y2], dim=1))))
        return out
