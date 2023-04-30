# Copyright 2021 Toyota Research Institute.  All rights reserved.
import functools
import itertools
import logging
import os
from collections import OrderedDict
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset

from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.structures.boxes import BoxMode

from tridet.data import collect_dataset_dicts
from tridet.structures.boxes3d import GenericBoxes3D
from tridet.structures.pose import Pose
from tridet.utils.coco import create_coco_format_cache
from tridet.utils.geometry import project_points3d
from tridet.utils.visualization import float_to_uint8_color

LOG = logging.getLogger(__name__)

VALID_CLASS_NAMES = ("Car", "Pedestrian", "Cyclist", "Van", "Truck")

COLORS = [float_to_uint8_color(clr) for clr in sns.color_palette("bright", n_colors=8)]
COLORMAP = OrderedDict({
    "Car": COLORS[2],  # green
    "Pedestrian": COLORS[1],  # orange
    "Cyclist": COLORS[0],  # blue
    "Van": COLORS[6],  # pink
    "Truck": COLORS[5],  # brown
    "Person_sitting": COLORS[4],  #  purple
    "Tram": COLORS[3],  # red
    "Misc": COLORS[7],  # gray
})

MV3D_SPLIT_KITTI_3D_REMAP = {
    "train": "training",
    "val": "training",
    "test": "testing",
    "overfit": "training",
    "trainval": "training",
}


class KITTIDepthDataset(Dataset):
    def __init__(self, kitti_raw_root, kitti_depth_root, split_path, class_names, mode):
        self.class_names = class_names
        if mode == 'test':
            print('Not implemented yet')
            self._split = []
            return
        with open(split_path) as _f:
            lines = _f.readlines()
        self._split = [line.rstrip("\n") for line in lines]
        
        self.kitti_raw_root = kitti_raw_root
        self.kitti_depth_root = kitti_depth_root
        self.calibration_table = self._parse_calibration_files()


    # TODO: change to calibration keyed by date and sensor
    def _parse_calibration_files(self):
        calibration_table = {}
        for date in os.listdir(self.kitti_raw_root):
            calib_raw = self.read_calib_file(os.path.join(self.kitti_raw_root, date, "calib_cam_to_cam.txt"))
            for sensor in ['image_02', 'image_03']:
                calibration_table[(date, sensor)] = np.reshape(calib_raw[sensor.replace('image', 'P_rect')], (3, 4))[:, :3]

        return calibration_table

    @staticmethod
    def read_calib_file(filepath):
        """
        Read in a calibration file and parse into a dictionary

        Parameters
        ----------
        filepath : String
            File path to read from

        Returns
        -------
        calib : Dict
            Dictionary with calibration values
        """
        data = {}

        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data


    def __len__(self):
        return len(self._split)

    def __getitem__(self, idx):
        sample_id = self._split[idx]
        sample = OrderedDict()
        sample.update(self._get_sample_data(sample_id))
        return sample

    def _get_sample_data(self, sample_id):
        date, folder, sensor, _, file_name = sample_id.split("/")
        file_name = file_name.split(".")[0]
        intrinsics = self.calibration_table[(date, sensor)]
        datum = {}
        datum['intrinsics'] = list(intrinsics.flatten())
        # Consistent with COCO format
        datum['file_name'] = os.path.join(self.kitti_raw_root, sample_id)
        I = Image.open(datum['file_name'])
        datum['width'], datum['height'] = I.width, I.height
        datum['image_id'] = f'{date}_{sensor}_{file_name}'
        datum['sample_id'] = datum['image_id']
        # Get Depth
        datum['depth_image_name'] = os.path.join(self.kitti_depth_root, folder, \
                                                'proj_depth',
                                                'groundtruth',
                                                sensor,
                                                f'{file_name}.png'
                                                )
        I = Image.open(datum['depth_image_name'])
        datum['depth_width'], datum['depth_height'] = I.width, I.height
        datum['annotations'] = []
        return datum


def build_kitti_depth_dataset(
    split_path, kitti_raw_root, kitti_depth_root, mode='train', class_names=VALID_CLASS_NAMES
):
    dataset = KITTIDepthDataset(kitti_raw_root, kitti_depth_root, split_path, class_names, mode=mode)
    dataset_dicts = collect_dataset_dicts(dataset)
    return dataset_dicts


def register_kitti_depth_metadata(dataset_name):
    metadata = MetadataCatalog.get(dataset_name)
    dataset_dicts = DatasetCatalog.get(dataset_name)
