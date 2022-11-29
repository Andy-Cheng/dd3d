from typing import List, Optional, Union
import torch.nn as nn

from tridet.modeling.backbone.omni_scripts.utils import make_divisible, val2list
from tridet.modeling.backbone.omni_scripts.ops import ConvLayer, MBV1Block, MBV2Block, FusedMBV2Block, ResidualBlock, SeqBackbone

__all__ = ["MixFusedMobileNetV2"]


class MixFusedMobileNetV2(SeqBackbone):
	def __init__(
		self,
		width_mult=1.0,
		channel_divisor=8,
		ks: Union[int, List[int], None] = None,
		expand_ratio: Union[int, List[int], None] = None,
		depth: Union[int, List[int], None] = None,
		stage_width_list: Optional[List[int]] = None,
		act_func=None,
		block_type_list: Optional[List[str]] = None,
		channel_att_list: Union[None, str, List[Optional[str]]] = None,
	):
		
		ks = val2list(ks or 3, 5)
		expand_ratio = val2list(expand_ratio or 6, 5)
		depth = val2list(depth, 5)
		act_func = act_func or "relu"
		block_type_list = block_type_list or ["fmb", "fmb", "fmb", "mb", "mb", "mb"]
		channel_att_list = val2list(channel_att_list, 5)
		
		block_configs = [
			# t, n, s
			[expand_ratio[0], depth[0] or 2, ks[0], 2],
			[expand_ratio[1], depth[1] or 3, ks[1], 2],
			[expand_ratio[2], depth[2] or 4, ks[2], 2],
			[expand_ratio[3], depth[3] or 3, ks[3], 1],
			[expand_ratio[4], depth[4] or 3, ks[4], 2],
		]
		
		stage_width_list = stage_width_list or [32, 16, 24, 32, 64, 96, 160]
		for i, w in enumerate(stage_width_list):
			stage_width_list[i] = make_divisible(w * width_mult, channel_divisor)
		
		# input stem
		input_stem = nn.Sequential(
			ConvLayer(3, stage_width_list[0], 3, 2, act_func=act_func, first_layer=True),
			(FusedMBV2Block if block_type_list[0] == "fmb" else MBV1Block)(
				stage_width_list[0],
				stage_width_list[1],
				kernel_size=3,
				stride=1,
				act_func=(act_func, None),
				**({"expand_ratio": 1} if block_type_list[0] == "fmb" else {}),
			),
		)
		
		# stages
		stages = []
		in_channels = stage_width_list[1]
		for (t, n, k, s), c, block_type, channel_att_type in zip(
			block_configs, stage_width_list[2:], block_type_list[1:], channel_att_list,
		):
			blocks = []
			for i in range(n):
				stride = s if i == 0 else 1
				mb_conv = (FusedMBV2Block if block_type == "fmb" else MBV2Block)(
					in_channels,
					c,
					k,
					stride,
					expand_ratio=t,
					act_func=(act_func, None) if block_type == "fmb" else (act_func, act_func, None),
				)
				if channel_att_type is None:
					channel_att = None
				elif channel_att_type.startswith("se"):
					raise NotImplementedError
				elif channel_att_type.startswith("ca"):
					raise NotImplementedError
				else:
					channel_att = None
				if channel_att is not None:
					if isinstance(mb_conv, FusedMBV2Block):
						mb_conv = nn.Sequential(
							mb_conv.spatial_conv,
							channel_att,
							mb_conv.point_conv,
						)
					else:
						mb_conv = nn.Sequential(
							mb_conv.inverted_conv,
							mb_conv.depth_conv,
							channel_att,
							mb_conv.point_conv
						)
				if i != 0:
					mb_conv = ResidualBlock(
						mb_conv,
						nn.Identity(),
					)
				blocks.append(mb_conv)
				in_channels = c
			stages.append(nn.Sequential(*blocks))
		super(MixFusedMobileNetV2, self).__init__(input_stem, stages)
