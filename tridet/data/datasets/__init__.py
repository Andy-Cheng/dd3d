# Copyright 2021 Toyota Research Institute.  All rights reserved.
import logging
import random
from functools import partial

from detectron2.data import DatasetCatalog

from tridet.data.datasets.kitti_3d import register_kitti_3d_datasets
from tridet.data.datasets.nuscenes import register_nuscenes_datasets
from tridet.data.datasets.kitti_depth import register_kitti_depth_datasets
from tridet.data.datasets.kradar import register_kradar_datasets
from tridet.data.datasets.cruw import register_cruw_datasets



def register_datasets(cfg):
    required_datasets = []
    if not getattr(cfg, 'NOT_REGISTER_TRAIN', False):
        required_datasets.append(cfg.DATASETS.TRAIN.NAME)
    required_datasets.append(cfg.DATASETS.TEST.NAME)
    dataset_names = []
    dataset_names.extend(register_kitti_3d_datasets(required_datasets, cfg))
    dataset_names.extend(register_nuscenes_datasets(required_datasets, cfg))
    dataset_names.extend(register_kitti_depth_datasets(required_datasets, cfg))
    dataset_names.extend(register_kradar_datasets(required_datasets, cfg))
    dataset_names.extend(register_cruw_datasets(required_datasets, cfg))
    
    if cfg.ONLY_REGISTER_DATASETS:
        for name in dataset_names:
            DatasetCatalog.get(name)
    return dataset_names


def random_sample_dataset_dicts(dataset_name, num_samples=10):
    dataset_dicts = DatasetCatalog.get(dataset_name)
    num_samples = min(num_samples, len(dataset_dicts))
    random.seed(42)
    if num_samples > 0:
        inds = random.sample(range(len(dataset_dicts)), k=num_samples)
    else:
        # Use all dataset items.
        inds = list(range(len(dataset_dicts)))
    samples = [dataset_dicts[i] for i in inds]
    return samples, inds
