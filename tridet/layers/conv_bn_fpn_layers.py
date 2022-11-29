from typing import List

import torch
from detectron2.layers import Conv2d, ShapeSpec
from torch import nn

from tridet.layers.normalization import get_norm
from tridet.layers.separable_conv2d import ACTIVATIONS
from tridet.modeling.dd3d.utils import get_fpn_out_channels


class ConvBnFpnLayers(nn.Module):
    """
    """
    def __init__(
        self,
        num_layers,
        input_shape,
        norm_kwargs={},
        kernel_size=3,
        activation='gelu',
        groups=1,
        extra_input_dim=0,
        use_input_dim=True,
        output_dim=None,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "'kernel_size' must be odd."
        self._input_shape = input_shape
        self._extra_input_dim = extra_input_dim
        num_levels = len(input_shape)
        channels = get_fpn_out_channels(input_shape)

        if not use_input_dim:
            assert output_dim is not None, "'output_dim' must be given, if 'use_input_dim=False'."
        input_dim = channels + extra_input_dim
        out_channels = input_dim if use_input_dim else output_dim
        self._out_channels = out_channels

        conv_layers = []
        for l in range(num_layers):
            in_channels = input_dim if l == 0 else out_channels
            # Build convolution layers
            conv_kwargs = dict(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=False,  # BN is applied manually in forward()
                norm=None,
                activation=None,  # activation is applied manually in forward()
                groups=groups
            )
            conv_layers.append(Conv2d(**conv_kwargs))
        self.conv_layers = nn.ModuleList(conv_layers)

        # Define a BN layer per each (level, layer).
        self.bn_layers = nn.ModuleList()
        norm_kwargs = norm_kwargs or {}
        for _ in range(num_levels):
            self.bn_layers.append(nn.ModuleList([get_norm('BN', out_channels, norm_kwargs) for _ in range(num_layers)]))

        # Activation
        self.act = ACTIVATIONS[activation]

        self.init_weights()

    def output_shape(self):
        return [
            ShapeSpec(channels=self._out_channels, height=x.height, width=x.width, stride=x.stride)
            for x in self._input_shape
        ]

    def init_weights(self):
        for conv in self.conv_layers:
            nn.init.kaiming_normal_(conv.weight)  # mode = 'fan_in'

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        out = []
        for level, _bn_layers in enumerate(self.bn_layers):  # iterating over first bn dim first makes TS happy
            x_level = x[level]
            for conv, bn in zip(self.conv_layers, _bn_layers):
                x_level = conv(x_level)
                x_level = bn(x_level)
                x_level = self.act(x_level)
            out.append(x_level)
        return out
