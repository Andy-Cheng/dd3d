import torch.nn as nn
from detectron2.layers import ShapeSpec

from tridet.modeling.backbone.omni_scripts.utils import make_divisible
from tridet.modeling.backbone.omni_scripts.fused_mb_nets import MixFusedMobileNetV2
from tridet.modeling.backbone.omni_scripts.fpn import FPN
from tridet.modeling.backbone.omni_scripts.ops import ConvLayer

__all__ = ["BackboneFPN", "build_feature_extractor_all_fuse"]


def build_feature_extractor_all_fuse(return_list=False, width_mult=1.0, depth_mult=1.0):

	stage_width_list = [32, 16, 32, 56, 104, 120, 320]
	depth_list = [2, 2, 5, 4, 3]
	for i, width in enumerate(stage_width_list):
		stage_width_list[i] = make_divisible(width * width_mult, 8)
	for i, depth in enumerate(depth_list):
		depth_list[i] = int(depth * depth_mult)
	backbone = MixFusedMobileNetV2(
		width_mult=1.0,
		ks=[3, 3, 3, 3, 3],
		expand_ratio=[4, 4, 4, 4, 4],
		depth=depth_list,  # 2, 3, 5, 5, 5
		block_type_list=["fmb", "fmb", "fmb", "fmb", "fmb", "fmb"],
		stage_width_list=stage_width_list,
		channel_att_list=[None, None, None, None, None],
		act_func="relu",
	)
	fpn_width_mult = 0.7
	output_width = make_divisible(128 * width_mult, 8)
	
	fpn = FPN(
		inputs=[
			("ex_stage2", stage_width_list[-1], make_divisible(stage_width_list[-1] * fpn_width_mult, 8), 128),
			("ex_stage1", stage_width_list[-1], make_divisible(stage_width_list[-1] * fpn_width_mult, 8), 64),
			("stage5", stage_width_list[-1], make_divisible(stage_width_list[-1] * fpn_width_mult, 8), 32),
			("stage4", stage_width_list[-2], make_divisible(stage_width_list[-2] * fpn_width_mult, 8), 16),
			("stage2", stage_width_list[-4], make_divisible(stage_width_list[-4] * fpn_width_mult, 8), 8),
		],
		input_mode="cat_conv",
		middle_config={
			"all": ["fmb_e@4_k@3", "fmb_e@4_k@3"],
			8: ["fmb_e@4_k@3"],
		},
		channel_att=None,
		prefix="fpn",
		act_func="relu",
		spp_size=[3, 5, 7],
		use_pan=True,
		output_width=output_width,
	)
	model = BackboneFPN(
		backbone, fpn,
		n_extra_stage=2, last_channels=stage_width_list[-1], act_func="relu", return_list=return_list,
	)
	return model


class BackboneFPN(nn.Module):
	def __init__(
		self, backbone: nn.Module, fpn: FPN, last_channels: int, act_func="relu", n_extra_stage=0,
		return_list=False,
	):
		super(BackboneFPN, self).__init__()
		self.backbone = backbone
		self.fpn = fpn
		self.extra_stage = nn.ModuleList([
			# PoolingLayer("avg", kernel_size=2, stride=2)
			ConvLayer(last_channels, last_channels, 3, 2, act_func=act_func)
			# FusedMBV2Block(last_channels, last_channels, 3, 2, expand_ratio=4, act_func=(act_func, None))
			for _ in range(n_extra_stage)
		])
		self.return_list = return_list
	
	@property
	def n_extra_stage(self):
		return len(self.extra_stage)

	@property
	def size_divisibility(self):
		return 32 * (2 ** self.n_extra_stage)
	
	def output_shape(self):
		out_list = []
		for i, (key, in_channel, mid_channel, stride) in enumerate(self.fpn.inputs):
			channels = self.fpn.output_width or mid_channel
			out_list.append((f"{self.fpn.prefix}_out{i + 1}", ShapeSpec(channels=channels, stride=stride)))
		out_list = out_list[::-1]
		out_dict = {}
		for key, shape in out_list:
			out_dict[key] = shape
		return out_dict
	
	def forward(self, x):
		feed_dict = self.backbone(x)
		x = feed_dict["output"]
		for i, extra_stage in enumerate(self.extra_stage):
			feed_dict[f"ex_stage{i + 1}"] = x = extra_stage(x)
		feed_dict = self.fpn(feed_dict)
		if self.return_list:
			out_list = [feed_dict[key] for key in self.output_shape()]
			return out_list
		else:
			return feed_dict
