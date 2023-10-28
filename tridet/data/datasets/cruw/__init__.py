import logging
import os
from functools import partial

from detectron2.data import DatasetCatalog

from tridet.data.datasets.cruw.build import build_cruw_dataset, build_cruw_image_dataset, register_cruw_metadata

LOG = logging.getLogger(__name__)

data_root_dir = '/mnt/nas_cruw/CRUW_2022' # todo: change to the cruw root
calib_root = '/mnt/nas_cruw/cruw_calibs/' # todo: change to the calib root
# label_path = '/mnt/disk1/CRUW/new_cruw2_03-28-2022/labels.json'
label_path = '/mnt/nas_cruw/data/Day_Night_all.json' #'/mnt/disk1/CRUW/new_cruw_all_to_2022_0203/labels.json'

# cruw image seq for prediction
seqs = ['2022_0217_1255', '2022_0217_1258', '2022_0217_1449']

daytime1_seqs = ['2021_1120_1616', '2021_1120_1618', '2021_1120_1619', '2022_0203_1428', '2022_0203_1439', '2022_0203_1441', '2022_0203_1443', '2022_0203_1445', '2022_0203_1512', '2022_0217_1232', '2022_0217_1251', '2022_0217_1307', '2022_0217_1322']
nighttime1_seqs = ['2021_1120_1616', '2021_1120_1618', '2021_1120_1619', '2021_1120_1632', '2021_1120_1634'] # ['2021_1120_1632', '2021_1120_1634']
daytime_nighttime_1_seqs = ['2021_1120_1616', '2021_1120_1618', '2021_1120_1619', '2022_0203_1428', '2022_0203_1439', '2022_0203_1441', '2022_0203_1443', '2022_0203_1445', '2022_0203_1512', \
    '2022_0217_1232', '2022_0217_1251', '2022_0217_1307', '2022_0217_1322', '2021_1120_1632', '2021_1120_1634']


'''
    2022_0217_1523 and 2022_0217_1525 may be removed for cross-check fine-tune due to different distribution, pedestrian unlabeled seqs are also removed
    since we only test on Car
'''
cross_check_unlabeled_seqs = ['2022_0217_1246', '2022_0217_1253', '2022_0217_1255',  '2022_0217_1313', '2022_0217_1439', '2022_0217_1523']

unimonocam_seq = ['2022_0203_1441']

DATASET_DICTS_BUILDER = {
    # "cruw_train": (build_cruw_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, label_path=label_path)),
    # "cruw_image": (build_cruw_image_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, seq_dirs=seqs)),
    # 'daytime_train': (build_cruw_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, label_path='/mnt/disk1/CRUW/daytime_seqs_to_0217_train.json')),
    # 'daytime_val': (build_cruw_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, label_path='/mnt/disk1/CRUW/daytime_seqs_to_0217_val.json')),
    # 'daytime_val_original': (build_cruw_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, label_path='/mnt/disk1/CRUW/daytime_seqs_to_0217_val.json', img_dir='left')),
    # 'nighttime_test': (build_cruw_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, label_path='/mnt/disk1/CRUW/night_time_1120_train.json', img_dir='left_rrdnet_seq')),
    # 'nighttime_test_original': (build_cruw_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, label_path='/mnt/disk1/CRUW/night_time_1120_train.json', img_dir='left')),
    'cross_check': (build_cruw_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, label_path='/mnt/nas_cruw/data/cross_check.json')),
    'daytime1_train': (build_cruw_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, label_path='/mnt/nas_cruw/data/Daytime1_train.json')),
    'daytime1_train_original': (build_cruw_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, label_path='/mnt/nas_cruw/data/Daytime1_train.json', img_dir='left')),
    'daytime1_val': (build_cruw_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, label_path='/mnt/nas_cruw/data/Daytime1_val.json')),
    'daytime1_val_original': (build_cruw_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, label_path='/mnt/nas_cruw/data/Daytime1_val.json', img_dir='left')),
    'nighttime1_train': (build_cruw_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, label_path='/mnt/nas_cruw/data/Nighttime1_train.json')),
    'nighttime1_val': (build_cruw_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, label_path='/mnt/nas_cruw/data/Nighttime1_val.json')),
    'nighttime1_val_original': (build_cruw_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, label_path='/mnt/nas_cruw/data/Nighttime1_val.json', img_dir='left')),
    # use the following to generate predictions for neurlps exps.
    "daytime1_image": (build_cruw_image_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, seq_dirs=daytime1_seqs)),
    "daytime1_image_original": (build_cruw_image_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, seq_dirs=daytime1_seqs, img_dir='left')),
    "nighttime1_image": (build_cruw_image_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, seq_dirs=nighttime1_seqs)),
    "nighttime1_image_original": (build_cruw_image_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, seq_dirs=nighttime1_seqs, img_dir='left')),
    "day_nighttime1_image": (build_cruw_image_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, seq_dirs=daytime_nighttime_1_seqs)),
    "day_nighttime1_image_original": (build_cruw_image_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, seq_dirs=daytime_nighttime_1_seqs, img_dir='left')),
    
    "cross_check_unlabeled_image": (build_cruw_image_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, seq_dirs=cross_check_unlabeled_seqs, img_dir='left', ignore_ratio=0.)),
    "unimonocam": (build_cruw_image_dataset, dict(root_dir=data_root_dir, calib_root=calib_root, seq_dirs=unimonocam_seq, img_dir='left', ignore_ratio=0.)),
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