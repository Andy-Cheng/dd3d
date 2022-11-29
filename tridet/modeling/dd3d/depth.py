import torch.nn as nn
import torch.nn.functional as F

from detectron2.layers import Conv2d
from tridet.utils.geometry import get_pixel_sizes_perspective_cams
from tridet.modeling.dd3d.utils import get_fpn_out_channels

class PacknetDepthHead(nn.Module):
    def __init__(
        self,
        net,
        input_shape,
        min_depth,
        max_depth,
        scale_depth_by_focal_length=None,  # NOTE: when use as depth-as-input, disable this and do the scaling online.
    ):
        super().__init__()

        self.net = net(input_shape=input_shape)

        input_shape = self.net.output_shape()
        in_channels = get_fpn_out_channels(input_shape)

        # Predictor
        conv_kwargs = dict(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            norm=None,
            activation=F.sigmoid
        )
        self.predictor = Conv2d(**conv_kwargs)

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.scale_depth_by_focal_length = scale_depth_by_focal_length

    def forward(self, x, cams):
        net_out = self.net(x)
        depth = [self.predictor(x) for x in net_out]

        if self.scale_depth_by_focal_length is not None:
            pixel_size = get_pixel_sizes_perspective_cams(cams)
            depth = [x / (pixel_size * self.scale_depth_by_focal_length).view(-1, 1, 1, 1) for x in depth]

        m, M = self.min_depth, self.max_depth
        depth = [(M - m) * x + m for x in depth]
        depth = [x.clamp(min=m, max=M) for x in depth]
        return {'depth': depth, 'depth_head_net_out': net_out}
