import logging
from functools import partial
from detectron2.data import DatasetCatalog
from tridet.data.datasets.kitti_depth.build import build_kitti_depth_dataset, register_kitti_depth_metadata

LOG = logging.getLogger(__name__)

DATASET_DICTS_BUILDER = {
    # Monocular datasets
    "kitti_depth": (build_kitti_depth_dataset, dict(split_path='/home/andy/ipl/dd3d/datasets_root/kitti_depth/filtered_eigen_zhou_files.txt', \
                                                    kitti_raw_root='/home/andy/ipl/dd3d/datasets_root/kitti_raw_sync',
                                                    kitti_depth_root='/home/andy/ipl/dd3d/datasets_root/kitti_depth/train',
                                                    )),
    "kitti_depth_test": (build_kitti_depth_dataset, dict(split_path='/home/andy/ipl/dd3d/datasets_root/kitti_depth/filtered_eigen_zhou_files.txt', \
                                                    kitti_raw_root='/home/andy/ipl/dd3d/datasets_root/kitti_raw_sync',
                                                    kitti_depth_root='/home/andy/ipl/dd3d/datasets_root/kitti_depth/train',
                                                    mode='test'
                                                    )),
}

METADATA_BUILDER = {name: (register_kitti_depth_metadata, {}) for name in DATASET_DICTS_BUILDER.keys()}


def register_kitti_depth_datasets(required_datasets, cfg):
    kitti_depth_datasets = sorted(list(set(required_datasets).intersection(DATASET_DICTS_BUILDER.keys())))
    if kitti_depth_datasets:
        LOG.info(f"KITTI-Depth dataset(s): {', '.join(kitti_depth_datasets)} ")
        for name in kitti_depth_datasets:
            fn, kwargs = DATASET_DICTS_BUILDER[name]
            DatasetCatalog.register(name, partial(fn, **kwargs))
            fn, kwargs = METADATA_BUILDER[name]
            fn(name, **kwargs)
    return kitti_depth_datasets
