import torch
from tridet.modeling.backbone.omni_scripts.backbone_with_fpn import build_feature_extractor_all_fuse

omninet_w10 = build_feature_extractor_all_fuse(
	return_list=False, width_mult=1.0, depth_mult=1.0,
)

checkpoint = torch.load(
	"omninet-small",
	map_location="cpu"
)
checkpoint = checkpoint["state_dict"]
omninet_w10.load_state_dict(checkpoint)
print(omninet_w10)
