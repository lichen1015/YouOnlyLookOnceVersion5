from torch import nn
import torch
from models.build_basic import build_conv_layer, build_norm_layer, build_activation_layer


class ConvBase(nn.Module):
    """每个卷积层都要 conv bn relu  写在主方法太麻烦了 看的我人都傻了"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 autopad=True,
                 norm=None,
                 act=dict(type='ReLU', inplace=True),
                 order=('conv', 'norm', 'act')):
        super().__init__()
        self.act = act
        self.with_norm = norm is not None
        self.with_activation = act is not None

        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias

        if autopad:
            padding = self.autopad(kernel_size, padding)
        elif padding is None:
            padding = 0

        self.conv = build_conv_layer(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm, norm_channels)
            self.add_module(self.norm_name, norm)

        # build activation layer
        if self.with_activation:
            act_cfg_ = act.copy()
            self.activate = build_activation_layer(act_cfg_)

        # Use msra init by default
        self.init_weights()

    def init_weights(self):
        if self.with_activation and self.act['type'] == 'LeakyReLU':
            nonlinearity = 'leaky_relu'
        else:
            nonlinearity = 'relu'
        nn.init.kaiming_uniform_(self.conv.weight, a=0, mode='fan_out', nonlinearity=nonlinearity)
        if self.with_norm:
            if hasattr(self.norm, 'weight') and self.norm.weight is not None:
                nn.init.constant_(self.norm.weight, 1)
            if hasattr(self.norm, 'bias') and self.norm.bias is not None:
                nn.init.constant_(self.norm.bias, self.norm)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x

    @staticmethod
    def autopad(k, p=None):
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        return p
