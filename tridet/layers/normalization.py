# Copyright 2021 Toyota Research Institute.  All rights reserved.
# Adapted from AdelaiDet
#   https://github.com/aim-uofa/AdelaiDet/
import logging
from functools import partial

import torch
from torch import nn

LOG = logging.getLogger(__name__)


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class Offset(nn.Module):
    def __init__(self, init_value=0.):
        super(Offset, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input + self.bias


class ModuleListDial(nn.ModuleList):
    def __init__(self, modules=None):
        super(ModuleListDial, self).__init__(modules)
        self.cur_position = 0

    def forward(self, x):
        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0
        return result

class DialableModules(nn.ModuleList):
    """
    Dialable modules. Typically used with hierarchical output from FPN feature extractors.
    Separate modules are applied to each FPN layer.
    """
    def __init__(self, modules=None):
        super(DialableModules, self).__init__(modules)
        self.cur_position = 0

    def forward(self, x):
        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0
        return result


class DialableBN(DialableModules):
    """
    Dialable batch-norm layers. Typical use case: all FPN layers shares a 2D convolutional decoder, but
    the batch-norm layers are not shared. That is, each FPN layers has its own shift and scale parameters, and keeps
    its own batch statistics (mean, scale).
    """
    def __init__(self, out_channels, num_bn_modules, **bn_kwargs):
        LOG.info(f"Initializing DialableBN with `num_bn_modules`={num_bn_modules}")
        bn_modules = [nn.BatchNorm2d(out_channels, **bn_kwargs) for _ in range(num_bn_modules)]
        super().__init__(bn_modules)

def get_norm(norm, out_channels, norm_kwargs={}):
    if not norm:
        return None

    norm_mapping = {
        "BN": nn.BatchNorm2d,
        "DialableBN": DialableBN,
        "GN": nn.GroupNorm,
    }

    norm_fn = partial(norm_mapping[norm], **norm_kwargs)
    if norm == "BN":
        return norm_fn(num_features=out_channels)
    elif norm == "DialableBN":
        return norm_fn(out_channels=out_channels)
    elif norm == "GN":
        return norm_fn(num_channels=out_channels)
