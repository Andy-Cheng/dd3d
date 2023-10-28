import os
from tqdm import tqdm
from tridet.utils.visualization import mosaic
from tridet.visualizers import box_3d_viz
import cv2
from collections import defaultdict, OrderedDict
import json
from tridet.data.datasets.cruw.build import CRUWDataset
from PIL import Image
from tridet.structures.pose import Pose
from detectron2.data import Metadata
import seaborn as sns
from tridet.utils.visualization import float_to_uint8_color
# from misc.write_video import write_video
from detectron2.data import detection_utils as d2_utils
import numpy as np
import pickle

COLORS = [float_to_uint8_color(clr) for clr in sns.color_palette("bright", n_colors=8)]

COLORMAP = OrderedDict({
    "Car": COLORS[2],  # green
    "Pedestrian": COLORS[1],  # orange
    "Cyclist": COLORS[0],  # blue
    "Van": COLORS[6],  # pink
    "Truck": COLORS[5],  # brown
    "Person_sitting": COLORS[4],  #  purple
    "Bus": COLORS[4],  #  purple
    "Tram": COLORS[3],  # red
    "Misc": COLORS[7],  # gray
})



def main():
    metadata = Metadata()
    metadata.thing_classes = ("Car", "Pedestrian", "Cyclist", "Van", "Truck", "Bus")
    metadata.thing_colors = [COLORMAP[klass] for klass in metadata.thing_classes]
    metadata.id_to_name = {idx: klass for idx, klass in enumerate(metadata.thing_classes)}
    metadata.contiguous_id_to_name = {idx: klass for idx, klass in enumerate(metadata.thing_classes)}
    metadata.name_to_contiguous_id = {name: idx for idx, name in metadata.contiguous_id_to_name.items()}
    cruw_dataset = CRUWDataset(cruw_root, calib_root, None, metadata.thing_classes, 'left_rrdnet_seq')
    
    if gt_path is not None:
        # gt: {seq: {frame_id: {info}}}
        # gt_tmp, seq_to_frame_counts = defaultdict(dict), dict()
        # with open(gt_path, 'r') as gt_file:
        #     gt_json = json.load(gt_file)['train']
        # print('Generating gt tmp\n')
        # for seq_frame in tqdm(gt_json):
        #     seq_name = seq_frame['seq_name']
        #     frame_name = seq_frame['frame_name']
        #     if seq_name not in seq_to_frame_counts:
        #         seq_to_frame_counts[seq_name] = len(os.listdir(os.path.join(cruw_root, seq_name, 'camera', 'left')))
        #     frame_info = {}
        #     frame_info['intrinsics'] = cruw_dataset._read_intrinsics(seq_name)
        #     img_dir = 'left_rrdnet_seq' # 'left' if seq_name in night_seq else 'left_rrdnet_seq'
        #     frame_info['file_name'] = os.path.join(cruw_root, seq_name, 'camera', img_dir, '{}.png'.format(frame_name))
        #     I = Image.open(frame_info['file_name'])
        #     frame_info['width'], frame_info['height'] = I.width, I.height
        #     frame_info['sample_id'] = '{}_{}'.format(seq_name, frame_name)
        #     frame_info['annotations'] = cruw_dataset.get_annotations(seq_frame, frame_info['sample_id'], frame_info['width'], frame_info['height'])
        #     extrinsics = Pose() # let left camera be the reference frame
        #     frame_info['extrinsics'] = {'wxyz': extrinsics.quat.elements.tolist(), 'tvec': extrinsics.tvec.tolist()}
        #     gt_tmp[seq_name].update({frame_name: frame_info})
            
        # # gt: {seq: [frame1_info, frame2_info, ...]} 
        # print('Generating gt \n')
        # gt = dict()
        # for seq_name, frame_counts in tqdm(seq_to_frame_counts.items()):
        #     frame_infos = []
        #     for i in range(frame_counts):
        #         frame_name = f'{i:06}'
        #         if frame_name in gt_tmp[seq_name]:
        #             frame_info = gt_tmp[seq_name][frame_name]
        #         else:
        #             frame_info = {}
        #             img_dir = 'left_rrdnet_seq' # 'left' if seq_name in night_seq else 'left_rrdnet_seq'
        #             frame_info['file_name'] = os.path.join(cruw_root, seq_name, 'camera', img_dir, '{}.png'.format(frame_name))
        #             I = Image.open(frame_info['file_name'])
        #             frame_info['width'], frame_info['height'] = I.width, I.height
        #             frame_info['sample_id'] = '{}_{}'.format(seq_name, frame_name)
        #             frame_info['annotations'] = []
        #             frame_info['intrinsics'] = cruw_dataset._read_intrinsics(seq_name)
        #             extrinsics = Pose() # let left camera be the reference frame
        #             frame_info['extrinsics'] = {'wxyz': extrinsics.quat.elements.tolist(), 'tvec': extrinsics.tvec.tolist()}
        #         frame_infos.append(frame_info)
        #     gt[seq_name] = frame_infos
        # with open('Day_Night_all_viz_format.pkl', 'wb') as f:
        #     pickle.dump(gt, f)
        with open(gt_path, 'rb') as f:
            gt = pickle.load(f)
    else:
        gt = None

    # pred: {seq: {frame_id: [instances]}}
    print('Generating pred \n')
    pred = dict()

    if pred_type == 'raw':
        pred_file = []
        for file in prediction_file_paths:
            with open(file, 'r') as json_file:
                pred_file.extend(json.load(json_file))
        for obj in tqdm(pred_file):
            seq_name = obj['image_id'][:14]
            frame_name = obj['image_id'][15:]
            if seq_name not in pred:
                pred[seq_name] = defaultdict(list)
            pred[seq_name][frame_name].append(obj)
    else:
        pred_file = {}
        for file in prediction_file_paths:
            with open(file, 'r') as json_file:
                pred_file.update(json.load(json_file))
        for seq_name, frames in pred_file.items():
            for frame_name, objs in frames.items():
                frame_name = f'{int(frame_name):06}'
                if seq_name not in pred:
                    pred[seq_name] = {}
                pred[seq_name][frame_name] = objs
    if gt is None:
        viz_pred_only(prediction_thresholds, pred, metadata, start_frame=Start_Frame, cruw_dataset = cruw_dataset)
    else:
        viz(prediction_thresholds, pred, gt, metadata, start_frame=Start_Frame)




def viz_pred_only(pred_thresholds, pred, metadata, start_frame=1260, cruw_dataset=None):
    for pred_threshold in pred_thresholds:
        print(f'Viz pred_threshold: {pred_threshold}')
        video_path = os.path.join(save_root_path, 'viz_pred_only', f'viz_threshold_{pred_threshold}', 'videos')
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        for seq_name, frame_infos in pred.items():
            # if seq_name != '2022_0217_1251': # debug
            #     continue
            print(f'Now visualize: {seq_name}')
            num_of_frames = len(os.listdir(os.path.join(cruw_root, seq_name, 'camera', 'left')))
            seq_root_path = os.path.join(save_root_path, 'viz_pred_only', f'viz_threshold_{pred_threshold}', seq_name)
            for i in tqdm(range(start_frame, num_of_frames)):
                frame_name = f'{i:06}'
                if not os.path.exists(seq_root_path):
                    os.makedirs(seq_root_path)
                if not frame_name in frame_infos:
                    pred_frame = []
                else:
                    pred_frame = frame_infos[frame_name]
                    # # debug
                    # for obj_index, obj in enumerate(pred_frame):
                    #     quat = [1, 0, 0, 0] #obj['bbox3d'][:4]
                    #     pred_frame[obj_index]['bbox3d'] = [*quat[1:], quat[0], *obj['bbox3d'][4:]] # set no rotation to debug
                image_info = {}
                image_info['extrinsics'] = Pose()
                image_info['intrinsics'] = cruw_dataset._read_intrinsics(seq_name)
                image_info['file_name'] = os.path.join(cruw_root, seq_name, 'camera', 'left_rrdnet_seq', '{}.png'.format(frame_name))
                viz_image = box_3d_viz(image_info, pred_frame, metadata, pred_threshold, viz_gt=False)
                imgs = list(viz_image.values())
                image = mosaic(imgs)
                cv2.imwrite(os.path.join(seq_root_path, f'{frame_name}.png'), image[:, :, ::-1])
            print('\n')
            # if save_video:
            #     write_video(seq_root_path, video_path, f'{seq_name}.mp4', fps=30) 

def viz(pred_thresholds, pred, gt, metadata, start_frame=1260):
    for pred_threshold in pred_thresholds:
        print(f'Viz pred_threshold: {pred_threshold}')
        video_path = os.path.join(save_root_path, 'viz', f'viz_threshold_{pred_threshold}', 'videos')
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        for seq_name, frame_infos in gt.items():
            if not seq_name in pred:
                continue
            print(f'Now visualize: {seq_name}')
            seq_root_path = os.path.join(save_root_path, 'viz', f'viz_threshold_{pred_threshold}', seq_name)
            for i in tqdm(range(start_frame, len(frame_infos))):
                frame_name = f'{i:06}'
                if not os.path.exists(seq_root_path):
                    os.makedirs(seq_root_path)
                if not frame_name in pred[seq_name]:
                    pred_frame = []
                else:
                    pred_frame = pred[seq_name][frame_name]
                    # # debug
                    # for obj_index, obj in enumerate(pred_frame):
                    #     quat = obj['bbox3d'][:4]
                    #     pred_frame[obj_index]['bbox3d'] = [*quat[1:], quat[0], *obj['bbox3d'][4:]] # set no rotation to debug

                viz_image = box_3d_viz(frame_infos[i], pred_frame, metadata, pred_threshold)
                imgs = list(viz_image.values())
                if viz_gt_2dbbox:         
                    # for debug purpose, draw 2d bbox
                    bbox_img = d2_utils.read_image(frame_infos[i]["file_name"], format='RGB')       
                    bboxes_2d = [ann['bbox'] for ann in frame_infos[i]['annotations']]
                    for bbox2d in bboxes_2d:
                        cv2.rectangle(
                        bbox_img, (int(bbox2d[0]), int(bbox2d[1])), (int(bbox2d[2]), int(bbox2d[3])),
                        (255, 0, 0),
                        thickness=2
                        )
                    imgs.append(bbox_img)
                image = mosaic(imgs)
                cv2.imwrite(os.path.join(seq_root_path, f'{frame_name}.png'), image[:, :, ::-1])
            print('\n')
            # if save_video:
            #     write_video(seq_root_path, video_path, f'{seq_name}.mp4', fps=30) 

            

if __name__ == '__main__':
    Start_Frame = 0 # 1260
    prediction_thresholds = [0.0] # , 0.4, 0.3, 0.2, 0.1, 0.
    save_video = False
    cruw_root = '/mnt/nas_cruw/CRUW_2022'
    calib_root = '/mnt/nas_cruw/cruw_calibs/'
    viz_gt_2dbbox = False
    
    # gt_path = '/mnt/nas_cruw/data/Day_Night_all.json'
    gt_path = 'Day_Night_all_viz_format.pkl'
    # gt_path = None
    pred_type = 'aggregate' # raw or 'aggregate'
    prediction_file_paths = ['/mnt/disk1/unimonocam/fs_10_1441/bbox3d_predictions_3dnms_0.0.json']
    save_root_path = os.path.split(prediction_file_paths[0])[0]
    main()
