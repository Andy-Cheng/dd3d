import json
from collections import defaultdict
import os
import numpy as np
import pickle
from scipy.spatial.transform import Rotation
import torch
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix
from pytorch3d.ops import box3d_overlap
from tqdm import tqdm


# each row: [quat with real part first, x, y, z, length, width, height, score, original_index]
# prediction: (N, )
def NMS3D_cruw(prediction, iou_th):
    if len(prediction) == 0:
        return prediction
    # sort prediction from high confidence to low confidence
    indices = torch.argsort(prediction[:, 10]).flip(dims=[0])
    prediction = prediction[indices, :]
    quats = prediction[:, :4]
    locs = prediction[:, 4:7]
    dims = prediction[:, 7:10]
    boxes = encode_box3d_quat(quats, dims, locs)
    boxes = torch.transpose(boxes, 1, 2)
    _, iou_3d = box3d_overlap(boxes, boxes)
    iou_3d = (torch.tril(iou_3d, diagonal=-1) > iou_th).to(torch.int)
    pick = (iou_3d.sum(axis=1) < 1).nonzero().flatten()

    return prediction[pick, -1]


def encode_box3d_quat(quats, dims, locs):
    device = quats.device
    N = quats.shape[0]
    R = quaternion_to_matrix(quats)
    dims = dims.reshape(-1, 1).repeat(1, 8) # [[eight w], [eight h], [eight l]]  # (N*3, 8)
    dims[::3, :4], dims[1::3, :4], dims[2::3, :4] = 0.5 * dims[::3, :4], 0.5 * dims[1::3, 4:], 0.5 * dims[2::3, :4]
    dims[::3, 4:], dims[1::3, 4:], dims[2::3, 4:] = -0.5 * dims[::3, 4:], -0.5 * dims[1::3, 4:], -0.5 * dims[2::3, 4:]
    index = torch.tensor([[4, 0, 1, 5, 6, 2, 3, 7],
                            [4, 5, 0, 1, 6, 7, 2, 3],
                            [4, 5, 6, 7, 0, 1, 2, 3]]).repeat(N, 1).to(device=device) # follow the order of box3d_overlap
    box_3d_object = torch.gather(dims, 1, index)
    box_3d = torch.matmul(R , box_3d_object.reshape(N, 3, -1) )
    box_3d += locs.unsqueeze(-1)
    return box_3d


combine = True
viz = True

tr_LR_tvec = [2.54, -0.3, -0.7]
saved_picke_name = 'kradar_dla34_3dnms_weather1.pkl'
if combine:
    with open('outputs/2024-02-14/23-05-14/inference/final/kradar_test_good_v3/bbox3d_predictions.json') as f:
        train_result = json.load(f)
    with open('outputs/2024-02-14/21-47-09/inference/final/kradar_test_good_v3/bbox3d_predictions.json') as f:
        test_result = json.load(f)

    result = train_result + test_result

    kradar_root = '/mnt/nas_kradar/kradar_dataset/dir_all'
    seq_camera_2_rdr_id = defaultdict(dict)
    for seq in os.listdir(kradar_root):
        label_files = os.listdir(os.path.join(kradar_root, seq, 'info_label'))
        for label_file in label_files:
            rdr_id, camera_id = label_file.split('_')
            camera_id = camera_id.split('.')[0]
            seq_camera_2_rdr_id[seq][camera_id] = rdr_id

    result_by_seq = {}

    for box in result:
        seq, camera_frame = box.pop('image_id').split('_')
        del box['file_name']
        if seq not in result_by_seq:
            result_by_seq[seq] = defaultdict(list)
        # rdr_frame = 
        bbox3d = box.pop('bbox3d')
        quat = bbox3d[:4]
        x, y, z = bbox3d[4:7]
        W, L, H = bbox3d[7:]
        r = Rotation.from_quat([*quat[1:], quat[0]]).as_euler('yxz')[0].item()
        z += 0.7
        x -= 0.1
        new_x, new_y, new_z = z-tr_LR_tvec[0], -x-tr_LR_tvec[1], -y-tr_LR_tvec[2]
        # if new_y < -30 or new_y > 30:
        #     print(new_y)
        # if new_x < 0 or new_x > 80 or new_y < -30 or new_y > 30 or new_z < -2 or new_z > 7.6:
        #     continue
        if new_x < 0 or new_x > 80 or new_y < -15 or new_y > 15 or new_z < -2 or new_z > 7.6:
            continue
        box['box3d'] = [new_x, new_y, new_z, L, W, H, float(r+np.pi/2)]
        box['quat'] = quat
        box['box3d_cam'] = bbox3d
        result_by_seq[seq][f'{camera_frame}/{seq_camera_2_rdr_id[seq][camera_frame]}'].append(box)

# def nms_filter_cruw(pred_path, prediction_format):
    iou_threshold = 0.0
    for seq_name, frames in result_by_seq.items():
        print(f'Now processing: {seq_name}')
        for frame_name, objs in tqdm(frames.items()):
            objects = []
            for i, obj in enumerate(objs):
                # each row: [quat with real part first, x, y, z, length, width, height, score, original_index]
                bbox3d = [*obj['box3d_cam'][:7], obj['box3d_cam'][8], obj['box3d_cam'][7], obj['box3d_cam'][9]] # ROI1 in Cenrad
                objects.append([*bbox3d, obj['score_3d'], float(i)])
            objects = torch.tensor(objects)
            pick_indexes = NMS3D_cruw(objects, iou_threshold).flatten().to(torch.int).tolist()
            result_by_seq[seq_name][frame_name] = [objs[pick_index] for pick_index in pick_indexes]
    save_file_name = f'dd3d_dla34_3dnms_{iou_threshold}_all.json'
    save_path = os.path.join(save_file_name)
    with open(save_path, 'w') as out_file:
        json.dump(result_by_seq, out_file, indent=2)



    result_by_seq_pkl = {}
    for seq, rdr_id_2_boxes in result_by_seq.items():
        for id, boxes in rdr_id_2_boxes.items():
            boxes_dict = {}
            box3d = []
            scores = []
            label_preds = []
            for box in boxes:
                box3d.append(box['box3d'])
                scores.append(box['score_3d'])
                label_preds.append(box['category_id'])
            boxes_dict['box3d'] = np.array(box3d).astype(np.float32)
            boxes_dict['scores'] = np.array(scores).astype(np.float32)
            boxes_dict['label_preds'] = np.array(label_preds).astype(np.int64)
            result_by_seq_pkl[f'{seq}/{id}'] = boxes_dict

    with open(saved_picke_name, 'wb') as f:
        pickle.dump(result_by_seq_pkl, f)


if viz:
    root = '/mnt/ssd3/kradar_cam_result'
    checkpoint_name = 'dd3d_dla34_3dnms_weather1'
    dataset_split = 'all'

    with open(saved_picke_name, 'rb') as f:
        pred = pickle.load(f)

    class_names = ['Sedan', 'BusorTruck']

    # TODO: make pred_new sorted by seq_name
    pred_viz_format = defaultdict(dict)
    for k, v in pred.items():
        seq, frame, rdr_frame = k.split('/')
        frame_objs, viz_frame_objs = [], []
        for i in range(len(v['box3d'])):
            frame_obj = {}
            frame_obj['obj_id'] = ''
            frame_obj['obj_type'] = class_names[int(v['label_preds'][i])]
            frame_obj['score'] = float(v['scores'][i])
            frame_obj['psr'] = {}
            x, y, z, l, w, h, theta = v['box3d'][i].tolist()
            frame_obj['psr']['position'] = [x, y, z]
            frame_obj['psr']['rotation'] = [0., 0., theta]
            frame_obj['psr']['scale'] = [l, w, h]
            frame_objs.append(frame_obj)
            frame_obj['psr']['position'] = {'x': x+tr_LR_tvec[0], 'y': y+tr_LR_tvec[1] , 'z': z+tr_LR_tvec[2] }
            frame_obj['psr']['rotation'] = {'x': 0, 'y': 0, 'z': theta}
            frame_obj['psr']['scale'] = {'x': l, 'y': w, 'z': h}
            viz_frame_objs.append(frame_obj)
        pred_viz_format[seq][frame] = {'objs': viz_frame_objs, 'split': dataset_split}
        
    save_pred_dir = os.path.join(root, f"{checkpoint_name}")
    os.makedirs(save_pred_dir, exist_ok=True)
    with open(os.path.join(save_pred_dir, f"{dataset_split}_prediction_viz_format.json"), "w") as f:
        json.dump(pred_viz_format, f, indent=2)