import logging
import os
from functools import partial

from detectron2.data import DatasetCatalog

from tridet.data.datasets.cruw.build import build_cruw_dataset, build_cruw_image_dataset, register_cruw_metadata

LOG = logging.getLogger(__name__)

data_root_dir = '/mnt/disk2/CRUW22_IMG' # todo: change to the cruw root
calib_root = '/mnt/ssd3/CRUW3D/cruw_calibs' # todo: change to the calib root

# cruw image seq for prediction
daytime1_seqs = ['2021_1120_1616', '2021_1120_1618', '2021_1120_1619', '2022_0203_1428', '2022_0203_1439', '2022_0203_1441', '2022_0203_1443', '2022_0203_1445', '2022_0203_1512', '2022_0217_1232', '2022_0217_1251', '2022_0217_1307', '2022_0217_1322']
nighttime1_seqs = ['2021_1120_1616', '2021_1120_1618', '2021_1120_1619', '2021_1120_1632', '2021_1120_1634'] # ['2021_1120_1632', '2021_1120_1634']
daytime_nighttime_1_seqs = ['2021_1120_1616', '2021_1120_1618', '2021_1120_1619', '2022_0203_1428', '2022_0203_1439', '2022_0203_1441', '2022_0203_1443', '2022_0203_1445', '2022_0203_1512', '2022_0217_1232', '2022_0217_1251', '2022_0217_1307', '2022_0217_1322', '2021_1120_1632', '2021_1120_1634']



DATASET_DICTS_BUILDER = {
    
    'carbus_train': (build_cruw_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, label_path='/mnt/ssd3/CRUW3D/cam_labels/CRUW3DCarTruck.json', split_type='train')),
    'carbus_test': (build_cruw_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, label_path='/mnt/ssd3/CRUW3D/cam_labels/CRUW3DCarTruck.json', split_type='test')),

    # 'daytime1_train': (build_cruw_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, label_path='/mnt/nas/nas_cruw/data/Daytime1_train.json')),
    # 'daytime1_val': (build_cruw_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, label_path='/mnt/nas/nas_cruw/data/Daytime1_val.json')),
    # 'daytime1_val_original': (build_cruw_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, label_path='/mnt/nas/nas_cruw/data/Daytime1_val.json', img_dir='left')),
    # 'nighttime1_train': (build_cruw_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, label_path='/mnt/nas/nas_cruw/data/Nighttime1_train.json')),
    # 'nighttime1_val': (build_cruw_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, label_path='/mnt/nas/nas_cruw/data/Nighttime1_val.json')),
    # 'nighttime1_val_original': (build_cruw_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, label_path='/mnt/nas/nas_cruw/data/Nighttime1_val.json', img_dir='left')),
    # # use the following to generate predictions for neurlps exps.
    # "daytime1_image": (build_cruw_image_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, seq_dirs=daytime1_seqs)),
    # "daytime1_image_original": (build_cruw_image_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, seq_dirs=daytime1_seqs, img_dir='left')),
    # "nighttime1_image": (build_cruw_image_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, seq_dirs=nighttime1_seqs)),
    # "nighttime1_image_original": (build_cruw_image_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, seq_dirs=nighttime1_seqs, img_dir='left')),
    
    # "day_nighttime1_image": (build_cruw_image_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, seq_dirs=daytime_nighttime_1_seqs)),
    # "day_nighttime1_image_original": (build_cruw_image_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, seq_dirs=daytime_nighttime_1_seqs, img_dir='left')),
}

METADATA_BUILDER = {name: (register_cruw_metadata, {}) for name in DATASET_DICTS_BUILDER.keys()}


def register_cruw_datasets(required_datasets, cfg):
    cruw_dataset = sorted(list(set(required_datasets).intersection(DATASET_DICTS_BUILDER.keys())))
    if cruw_dataset:
        LOG.info(f"CRUW dataset(s): {', '.join(cruw_dataset)} ")
        for name in cruw_dataset:
            fn, kwargs = DATASET_DICTS_BUILDER[name]
            DatasetCatalog.register(name, partial(fn, **kwargs))

            fn, kwargs = METADATA_BUILDER[name]
            kwargs.update({'coco_cache_dir': cfg.TMP_DIR})
            fn(name, **kwargs)
    return cruw_dataset

# def register_cruw_datasets_no_cfg(required_datasets):
#     cruw_dataset = sorted(list(set(required_datasets).intersection(DATASET_DICTS_BUILDER.keys())))
#     if cruw_dataset:
#         LOG.info(f"CRUW dataset(s): {', '.join(cruw_dataset)} ")
#         for name in cruw_dataset:
#             fn, kwargs = DATASET_DICTS_BUILDER[name]
#             DatasetCatalog.register(name, partial(fn, **kwargs))

#             fn, kwargs = METADATA_BUILDER[name]
#             fn(name, **kwargs)
#     return cruw_dataset