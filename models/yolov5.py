import torch
from torch import nn
from component import SimBottleneckCSP, SPP, Focus
from models.common import ConvBase


class Yolov5(nn.Module):
    """
        YOlov5 代码复现 version1版本 架构随便乱填的

    """

    def __init__(self, num_class=80, in_channels=3, anchor=None):
        super().__init__()
        if anchor is None:
            anchor = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]

        # 这边应该需要几个超参数
        cd = 2  # channels_divide
        wd = 3  # 块数整除
        # backbone部分
        # 1 Focus 头 （需要改进，或者直接去掉）
        self.focus = Focus(in_channels, 64 // cd)
        # 2 convBase  CSP1
        self.conv1 = ConvBase(64 // cd, 128 // cd, 3, 2)
        self.csp1 = SimBottleneckCSP(128 // cd, 128 // cd, n=3 // wd)
        # 3 conv2 csp2
        self.conv2 = ConvBase(128 // cd, 256 // cd, 3, 2)
        self.csp2 = SimBottleneckCSP(256 // cd, 256 // cd, n=9 // wd)
        # 4 conv3 csp3
        self.conv3 = ConvBase(256 // cd, 512 // cd, 3, 2)
        self.csp3 = SimBottleneckCSP(512 // cd, 512 // cd, n=9 // wd)
        # 5 conv4 spp csp4
        self.conv4 = ConvBase(512 // cd, 1024 // cd, 3, 2)
        self.spp = SPP(1024 // cd, 1024 // cd)
        self.csp4 = SimBottleneckCSP(1024 // cd, 1024 // cd, n=3 // wd)
        # =============================
        # Head部分
        # 6 conv5 up1 csp5
        self.conv5 = ConvBase(1024 // cd, 512 // cd, 3, 2)
        self.up1 = nn.Upsample(scale_factor=2)
        self.csp5 = SimBottleneckCSP(1024 // cd, 512 // cd, n=3 // wd, shortcut=False)
        # 7 conv6 up2 csp6     x_small
        self.conv6 = ConvBase(512 // cd, 256 // cd, 3, 2)
        self.up2 = nn.Upsample(scale_factor=2)
        self.csp6 = SimBottleneckCSP(512 // cd, 256 // cd, n=3 // wd, shortcut=False)
        # 8 conv7 csp7  x_medium
        self.conv7 = ConvBase(256 // cd, 256 // cd, 3, 2)
        self.csp7 = SimBottleneckCSP(512 // cd, 512 // cd, n=3 // wd, shortcut=False)
        # 9 conv8 csp8  x_large
        self.conv8 = ConvBase(512 // cd, 512 // cd, 3, 2)
        self.csp8 = SimBottleneckCSP(512 // cd, 1024 // cd, n=3 // wd, shortcut=False)

    def _build_backbone(self, x):
        x = self.focus(x)
        x = self.conv1(x)
        x = self.csp1(x)
        x_p3 = self.conv2(x)  # p3
        x = self.csp2(x_p3)

        x_p4 = self.conv3(x)  # p4
        x = self.csp3(x_p4)

        x_p5 = self.conv4(x)  # p5
        x = self.spp(x_p5)
        x = self.csp4(x)

        return x_p3, x_p4, x_p5, x

    def _build_head(self, p3, p4, p5, feat):
        h_p5 = self.conv5(feat)
        x = self.up1(h_p5)
        x_concat = self.csp5(torch.cat([x, p4]), dim=1)
        x = self.csp5(x_concat)

        h_p4 = self.conv6(x)
        x = self.up2(h_p4)
        x_concat = torch.concat([x, p3], dim=1)
        x_small = self.csp6(x_concat)

        x = self.conv7(x_small)
        x_concat = torch.concat([x, p4], dim=1)
        x_medium = self.csp7(x_concat)

        x = self.conv8(x_medium)
        x_concat = torch.cat([x, h_p5], dim=1)
        x_large = self.csp8(x_concat)

        return x_small, x_medium, x_large

    def forward(self, x):
        p3, p4, p5, feat = self._build_backbone(x)
        xs, xm, xl = self._build_head(p3, p4, p5, feat)
        return xs, xm, xl
