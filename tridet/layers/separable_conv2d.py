import math

import torch
from detectron2.layers import Conv2d
from torch import nn
from torch.nn import functional as F
from torch.nn.init import _calculate_correct_fan, calculate_gain

from tridet.layers.normalization import get_norm

ACTIVATIONS = {
    'relu': F.relu,
    'gelu': F.gelu,
}


def kaiming_uniform_groups_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu', groups=1):
    """'torch.nn.init.kaiming_uniform_()' with 'groups'.

    If 'mode=="fan_out"', fan is divided by 'groups', yielding larger std of weights.
    """
    if 0 in tensor.shape:
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    if mode == 'fan_out':
        fan //= groups
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def kaiming_normal_groups_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu', groups=1):
    """'torch.nn.init.kaiming_normal_()' with 'groups'.

    If 'mode=="fan_out"', fan is divided by 'groups', yielding larger std of weights.
    """
    if 0 in tensor.shape:
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    if mode == 'fan_out':
        fan //= groups
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)


class SeparableConv2d(nn.Module):
    """ Separable Conv
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        bias=None,
        channel_multiplier=1.0,
        num_in_channels_per_group=1,  # depth-separable conv.
        norm='BN',
        norm_kwargs={},
        activation=None,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd."
        assert in_channels % num_in_channels_per_group == 0, "'in_channels' must be divisible by 'num_in_channels_per_group'"
        hidden_channels = int(in_channels * channel_multiplier)
        groups = in_channels // num_in_channels_per_group
        self.conv_dw = Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            dilation=dilation,
            bias=False,
            norm=None,
            activation=None,
            groups=groups
        )

        norm_kwargs = norm_kwargs or {}
        norm_layer = get_norm(norm, hidden_channels, norm_kwargs=norm_kwargs) if isinstance(norm, str) else norm
        if bias is None:
            bias = norm_layer is None
        act = ACTIVATIONS[activation] if isinstance(activation, str) else activation
        self.conv_pw = Conv2d(
            hidden_channels, out_channels, kernel_size=1, stride=1, bias=bias, norm=norm_layer, activation=act
        )

        self.groups = in_channels
        self.init_weights()

    def init_weights(self):
        # This seems important to make the network output roughly zero-mean, unit-std.
        kaiming_normal_groups_(self.conv_dw.weight, mode='fan_out', nonlinearity='linear', groups=self.groups)
        kaiming_normal_groups_(self.conv_pw.weight, mode='fan_out', nonlinearity='relu', groups=1)
        if self.conv_pw.bias is not None:
            nn.init.constant_(self.conv_pw.bias, 0)

    def forward(self, x):
        return self.conv_pw(self.conv_dw(x))
