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

import json
from math import ceil

LOG = logging.getLogger(__name__)

# VALID_CLASS_NAMES = ("Car", "Pedestrian")
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

# the dataset class from json file
class CRUWDataset(Dataset):
    # root_dir: CRUW root dir, label_path: json label path
    def __init__(self, root_dir, calib_root, label_path, class_names, image_dir, split_type='train', box2d_from_box3d=True):
        self.root_dir = root_dir
        self.calib_root = calib_root
        self.class_names = class_names
        self._name_to_id = {name: idx for idx, name in enumerate(class_names)}
        self._box2d_from_box3d = box2d_from_box3d
        self._image_dir = image_dir
        # read in json file
        if label_path is None:
            self.samples = []
        else:
            with open(label_path, 'r') as json_file:
                self.samples = json.load(json_file)[split_type]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        datum = {}
        sample = self.samples[idx]
        datum['intrinsics'] = self._read_intrinsics(sample['seq_name'])
        datum['file_name'] = os.path.join(self.root_dir, sample['seq_name'], 'camera', self._image_dir, '{}.png'.format(sample['frame_name']))
        I = Image.open(datum['file_name'])
        datum['width'], datum['height'] = I.width, I.height
        datum['image_id'] = '{}_{}'.format(sample['seq_name'], sample['frame_name'])
        datum['sample_id'] = datum['image_id']
        datum['annotations'] = self.get_annotations(sample, datum['sample_id'], datum['width'], datum['height'])
        extrinsics = Pose() # let left camera be the reference frame
        datum['extrinsics'] = {'wxyz': extrinsics.quat.elements.tolist(), 'tvec': extrinsics.tvec.tolist()}
        return datum

    def _read_intrinsics(self, seq_name):
        calib_file_path = os.path.join(self.calib_root, seq_name, 'calib', 'camera', 'left.json')
        with open(calib_file_path, 'r') as calib_file:
            K = np.array(json.load(calib_file)['intrinsic']).reshape(3, 4)[:3, :3].flatten().tolist()
        return K

    def get_annotations(self, sample, sample_id, width, height):
        annotations = []
        for idx, object in enumerate(sample['objs']):
            class_name = object['obj_type']
            if class_name not in self.class_names:
                continue
            annotation = OrderedDict(category_id=self._name_to_id[class_name], instance_id=f'{sample_id}_{idx}')
            annotation.update(self._get_3d_annotation(object))
            if self._box2d_from_box3d:
                intrinsics = np.array(self._read_intrinsics(sample['seq_name']), dtype=np.float64).reshape(3, 3)
                annotation.update(self._compute_box2d_from_box3d(annotation['bbox3d'], intrinsics, width, height))
            else:
                l, t, w, h = object['2dbbox']
                annotation.update(OrderedDict(bbox=[l, t, l+w, t+h], bbox_mode=BoxMode.XYXY_ABS))
            annotations.append(annotation)
        return annotations

    def _get_3d_annotation(self, object):
        """Convert annotation data frame to 3D bounding box annotations.
        Labels are provided in the reference frame of left camera.
        """
        length, width, height = object['scale']
        x, y, z = object['position'] # 3d box center's position
        quat = object['quat'] # quaternion

        box_pose = Pose(
            wxyz=Quaternion(quat[3], *quat[:3]),
            tvec=np.float64([x, y, z])
        )

        box3d = GenericBoxes3D(box_pose.quat.elements, box_pose.tvec, [width, length, height])
        vec = box3d.vectorize().tolist()[0]
        distance = float(np.linalg.norm(vec[4:7]))

        return OrderedDict([('bbox3d', vec), ('distance', distance)])

    def _get_2d_annotation(self, label):
        l, t, r, b = label[4:8]
        return OrderedDict(bbox=[l, t, r, b], bbox_mode=BoxMode.XYXY_ABS)

    def _compute_box2d_from_box3d(self, box3d, K, width, height):
        box = GenericBoxes3D(box3d[:4], box3d[4:7], box3d[7:])
        corners = project_points3d(box.corners.cpu().numpy()[0], K)
        corners_x = np.clip(corners[:, 0], 0, float(width-1))
        corners_y = np.clip(corners[:, 1], 0, float(height-1))

        l, t = corners_x.min(), corners_y.min()
        r, b = corners_x.max(), corners_y.max()
        return OrderedDict(bbox=[l, t, r, b], bbox_mode=BoxMode.XYXY_ABS)

# the dataset class from image folder
class CRUWImageDataset(Dataset):
    # root_dir: CRUW root dir
    def __init__(self, root_dir, calib_root, seqs, class_names, image_dir, ignore_ratio=0.7):
        self.root_dir = root_dir
        self.calib_root = calib_root
        self.class_names = class_names
        self._name_to_id = {name: idx for idx, name in enumerate(class_names)}
        self._image_dir = image_dir
        samples = []

        for seq in seqs:
            seq_path = os.path.join(root_dir, seq, 'camera', image_dir)

            img_names = sorted(os.listdir(seq_path))
            img_names = img_names[ceil(ignore_ratio * len(img_names)):]
            print('first img in {}: {}'.format(seq, img_names[0]))            
            for img_name in img_names:
                img_name = img_name.split('.')[0]
                sample_id = f'{seq}-{img_name}'
                samples.append(sample_id)
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        datum = {}
        sample = self.samples[idx]
        seq_name, frame_name = sample.split('-')
        datum['intrinsics'] = self._read_intrinsics(seq_name)
        datum['file_name'] = os.path.join(self.root_dir, seq_name, 'camera', self._image_dir, '{}.png'.format(frame_name))
        I = Image.open(datum['file_name'])
        datum['width'], datum['height'] = I.width, I.height
        datum['image_id'] = sample
        datum['sample_id'] = sample
        return datum

    def _read_intrinsics(self, seq_name):
        calib_file_path = os.path.join(self.calib_root, seq_name, 'calib', 'camera', 'left.json')
        with open(calib_file_path, 'r') as calib_file:
            K = np.array(json.load(calib_file)['intrinsic']).reshape(3, 4)[:3, :3].flatten().tolist()
        return K

# @functools.lru_cache(maxsize=1000)
def build_cruw_dataset(
    root_dir, calib_root, label_path, class_names=VALID_CLASS_NAMES, img_dir='left', split_type='train'
):
    dataset = CRUWDataset(root_dir, calib_root, label_path, class_names, split_type=split_type, image_dir=img_dir)
    dataset_dicts = collect_dataset_dicts(dataset)
    return dataset_dicts

def build_cruw_image_dataset(
    root_dir, calib_root, seq_dirs, class_names=VALID_CLASS_NAMES, img_dir='left', ignore_ratio=0.7
):
    dataset = CRUWImageDataset(root_dir, calib_root, seq_dirs, class_names, image_dir=img_dir, ignore_ratio=ignore_ratio)
    dataset_dicts = collect_dataset_dicts(dataset)
    return dataset_dicts

def register_cruw_metadata(dataset_name, valid_class_names=VALID_CLASS_NAMES, coco_cache_dir='/mnt/disk1/tmp/dd3d'):
    metadata = MetadataCatalog.get(dataset_name)
    metadata.thing_classes = valid_class_names
    metadata.thing_colors = [COLORMAP[klass] for klass in metadata.thing_classes]

    metadata.id_to_name = {idx: klass for idx, klass in enumerate(metadata.thing_classes)}
    metadata.contiguous_id_to_name = {idx: klass for idx, klass in enumerate(metadata.thing_classes)}
    metadata.name_to_contiguous_id = {name: idx for idx, name in metadata.contiguous_id_to_name.items()}

    # dataset_dicts = DatasetCatalog.get(dataset_name)
    # metadata.json_file = create_coco_format_cache(dataset_dicts, metadata, dataset_name, coco_cache_dir)
    # LOG.info(f'COCO json file: {metadata.json_file}')

    metadata.evaluators = ["kitti3d_evaluator"] # "coco_evaluator", 
    metadata.pred_visualizers = ["box3d_visualizer"] # "d2_visualizer",
    metadata.loader_visualizers = ["box3d_visualizer"] # "d2_visualizer", 
