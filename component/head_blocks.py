from torch import nn
import torch
from utils import autopad


class SPP(nn.Module):
    """YOLOv3-SPP"""
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        # 随便取的隐藏层通道数
        c_ = c1 // 2

        # BaseBlock
        self.conv1 = nn.Conv2d(c1, c_, 1, 1, autopad(1), groups=1)
        self.bn1 = nn.BatchNorm2d(c_)
        self.relu = nn.ReLU(True)

        # maxpool
        self.m = nn.ModuleList([nn.MaxPool2d(x, stride=1, padding=x // 2) for x in k])

        # ConvBase2
        self.conv2 = nn.Conv2d(c_ * (len(k) + 1), c2, 1, 1, autopad(1), groups=1)
        self.bn2 = nn.BatchNorm2d(c2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        out = torch.cat([x] + [_m(x) for _m in self.m], dim=1)
        return self.relu(self.bn2(self.conv2(out)))
