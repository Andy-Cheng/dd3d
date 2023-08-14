import logging
import os
from functools import partial

from detectron2.data import DatasetCatalog

from tridet.data.datasets.kradar.build import build_kradar_dataset, register_kradar_metadata

LOG = logging.getLogger(__name__)
KRADAR_ROOT = 'kradar'
KRADAR_LABEL_ROOT = 'kradar_label'

DATASET_DICTS_BUILDER = {
    'kradar_train_good_v3': (build_kradar_dataset, dict(label_path='kradar_cam_aligned_v2.json', split_type='train')),
    'kradar_test_good_v3': (build_kradar_dataset, dict(label_path='kradar_cam_aligned_v2.json', split_type='test')),
}

METADATA_BUILDER = {name: (register_kradar_metadata, {}) for name in DATASET_DICTS_BUILDER.keys()}


def register_kradar_datasets(required_datasets, cfg):
    kradar_dataset = sorted(list(set(required_datasets).intersection(DATASET_DICTS_BUILDER.keys())))
    if kradar_dataset:
        LOG.info(f"K-Radar dataset(s): {', '.join(kradar_dataset)} ")
        for name in kradar_dataset:
            fn, kwargs = DATASET_DICTS_BUILDER[name]
            kwargs.update({'root_dir': os.path.join(cfg.DATASET_ROOT, KRADAR_ROOT)})
            label_path = kwargs['label_path']
            kwargs.update({'label_path': os.path.join(cfg.DATASET_ROOT, KRADAR_LABEL_ROOT, label_path)})
            DatasetCatalog.register(name, partial(fn, **kwargs))
            fn, kwargs = METADATA_BUILDER[name]
            kwargs.update({'coco_cache_dir': cfg.TMP_DIR})
            fn(name, **kwargs)
    return kradar_dataset
