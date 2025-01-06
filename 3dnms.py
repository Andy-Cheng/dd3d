import os
from tqdm import tqdm
import csv
import shutil
import torch
from pytorch3d.ops import box3d_overlap
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix
from collections import defaultdict
import json
from detectron2.layers import batched_nms
import pickle

def check_last_line_break(predict_txt):
    f = open(predict_txt, 'rb+')
    try:
        f.seek(-1, os.SEEK_END)
    except:
        pass
    else:
        if f.__next__() == b'\n':
            f.seek(-1, os.SEEK_END)
            f.truncate()
    f.close()

def rady_to_matrix(rotys, N):
    device = rotys.device

    cos, sin = rotys.cos(), rotys.sin()

    i_temp = torch.tensor([[1, 0, 1],
                            [0, 1, 0],
                            [-1, 0, 1]]).to(dtype=torch.float32,
                                            device=device)
    ry = i_temp.repeat(N, 1).reshape(N, -1, 3)

    ry[:, 0, 0] *= cos
    ry[:, 0, 2] *= sin
    ry[:, 2, 0] *= sin
    ry[:, 2, 2] *= cos

    return ry

def radz_to_matrix(rotz, N):
    device = rotz.device

    cos, sin = rotz.cos(), rotz.sin()

    i_temp = torch.tensor([[1, -1, 0],
                            [1, 1, 0],
                            [0, 0, 1]]).to(dtype=torch.float32,
                                            device=device)
    rz = i_temp.repeat(N, 1).reshape(N, -1, 3)

    rz[:, 0, 0] *= cos
    rz[:, 0, 1] *= sin
    rz[:, 1, 0] *= sin
    rz[:, 1, 1] *= cos

    return rz

def radx_to_matrix(rotx, N):
    device = rotx.device

    cos, sin = rotx.cos(), rotx.sin()

    i_temp = torch.tensor([[1, 0, 0],
                            [0, 1, -1],
                            [0, 1, 1]]).to(dtype=torch.float32,
                                            device=device)
    rx = i_temp.repeat(N, 1).reshape(N, -1, 3)

    rx[:, 1, 1] *= cos
    rx[:, 1, 2] *= sin
    rx[:, 2, 1] *= sin
    rx[:, 2, 2] *= cos

    return rx

def encode_box3d(eulers, dims, locs):
    eulers = eulers.reshape(-1, 3)
    dims = dims.reshape(-1, 3)
    locs = locs.reshape(-1, 3)
    device = eulers.device
    N = eulers.shape[0]
    rx = radx_to_matrix(eulers[:, 0], N) # (N, 3, 3)
    ry = rady_to_matrix(eulers[:, 1], N) # (N, 3, 3)
    rz = radz_to_matrix(eulers[:, 2], N) # (N, 3, 3)
    dims = dims.reshape(-1, 1).repeat(1, 8) # [[eight w], [eight h], [eight l]]  # (N*3, 8)
    dims[::3, :4], dims[1::3, :4], dims[2::3, :4] = 0.5 * dims[::3, :4], 0.5 * dims[1::3, 4:], 0.5 * dims[2::3, :4]
    dims[::3, 4:], dims[1::3, 4:], dims[2::3, 4:] = -0.5 * dims[::3, 4:], -0.5 * dims[1::3, 4:], -0.5 * dims[2::3, 4:]
    index = torch.tensor([[4, 0, 1, 5, 6, 2, 3, 7],
                            [4, 5, 0, 1, 6, 7, 2, 3],
                            [4, 5, 6, 7, 0, 1, 2, 3]]).repeat(N, 1).to(device=device) # follow the order of box3d_overlap
    box_3d_object = torch.gather(dims, 1, index)
    box_3d = torch.matmul(torch.matmul(rz, torch.matmul(rx,ry)) , box_3d_object.reshape(N, 3, -1) )
    box_3d += locs.unsqueeze(-1)

    return box_3d

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

# prediction: (N, )
def NMS3D_kitti(prediction, iou_th):
    if len(prediction) == 0:
        return prediction
    # sort prediction from high confidence to low confidence
    indices = torch.argsort(prediction[:, -1]).flip(dims=[0])
    prediction = prediction[indices, :]
    dims = prediction[:, 4:7].roll(shifts=1, dims=1)
    eulers = prediction[:, 14:17]
    locs = prediction[:, 7:10]
    boxes = encode_box3d(eulers, dims, locs)
    boxes = torch.transpose(boxes, 1, 2)
    _, iou_3d = box3d_overlap(boxes, boxes)
    iou_3d = (torch.tril(iou_3d, diagonal=-1) > iou_th).to(torch.int)
    pick = (iou_3d.sum(axis=1) < 1).nonzero().flatten()

    return prediction[pick, :]


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

# for kitti annotation format
def nms_filter(pred_folder, save_dir):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    pred_files = os.listdir(pred_folder)
    for index in tqdm(range(len(pred_files))):
        file_name = pred_files[index]
        prediction_file = os.path.join(pred_folder, file_name)
        objs = [] 
        with open(prediction_file, 'r') as p_f:
            for line in p_f:
                line = line.strip()
                line_info = line.split(' ')[3:]
                objs.append([float(num) for num in line_info])
        objs = torch.tensor(objs)
        # l1 = len(objs)
        objs = NMS3D_kitti(objs, iou_threshold)
        # l2 = len(objs)
        # if l1 != l2:
        #     print('original objs: {}'.format(l1))
        #     print('after objs: {}'.format(l2), '\n')

        save_file = os.path.join(save_dir, file_name)
        with open(save_file, 'w', newline='') as f:
            w = csv.writer(f, delimiter=' ', lineterminator='\n')
            if len(objs) == 0:
                w.writerow([])
            else:
                for p in objs:
                    row = ['Car', 0, 0] + p.tolist()
                    w.writerow(row)
        check_last_line_break(save_file)



# transform [{single object in a frame}] to {seq_name: {frame_name: [obj]}}
def load_pred(file_path, prediction_type):
    predictions = dict()
    if prediction_type == 'raw':
        raw_predictions = []
        with open(file_path, 'r') as json_file:
            raw_predictions.extend(json.load(json_file))
        for obj in raw_predictions:
            seq_name = obj['image_id'][:14]
            frame_name = obj['image_id'][17:]
            if seq_name not in predictions:
                predictions[seq_name] = defaultdict(list)
            predictions[seq_name][frame_name].append(obj)

    else:
        with open(file_path, 'r') as json_file:
            predictions = json.load(json_file)
    return predictions


# for cruw annotation format
# each bbox3d in annotation file: [quat with real part first, width, length, height]
def nms_filter_cruw(pred_path, prediction_format):
    predictions = load_pred(pred_path, prediction_format)
    if enable_filter:
        for seq_name, frames in predictions.items():
            print(f'Now processing: {seq_name}')
            for frame_name, objs in tqdm(frames.items()):
                objects = []
                for i, obj in enumerate(objs):
                    bbox3d = [*obj['bbox3d'][:7], obj['bbox3d'][8], obj['bbox3d'][7], obj['bbox3d'][9]]
                    objects.append([*bbox3d, obj['score_3d'], float(i)])
                objects = torch.tensor(objects) 
                pick_indexes = NMS3D_cruw(objects, iou_threshold).flatten().to(torch.int).tolist()
                predictions[seq_name][frame_name] = [objs[pick_index] for pick_index in pick_indexes]
    dir, file_name = os.path.split(pred_path)
    file_name = file_name.split('.')[0]
    save_file_name = f'{file_name}_3dnms_{iou_threshold}.json' if enable_filter else f'{file_name}.json'
    save_path = os.path.join(dir, save_file_name)
    with open(save_path, 'w') as out_file:
        json.dump(predictions, out_file, indent=2)


# 2d nms filter
def nms2d_filter_cruw(pred_path, prediction_format):
    predictions = load_pred(pred_path, prediction_format)
    for seq_name, frames in predictions.items():
        print(f'Now processing: {seq_name}')
        for frame_name, objs in tqdm(frames.items()):
            n_obj = len(objs)
            bboxes = torch.zeros((n_obj, 4)).to('cuda')
            idxs = torch.zeros((n_obj,), dtype=torch.int).to('cuda')
            scores = torch.zeros((n_obj,)).to('cuda')
            for dt_id, dt in enumerate(objs):
                bboxes[dt_id] = torch.Tensor(dt['bbox'])
                bboxes[dt_id][2] += bboxes[dt_id][0]
                bboxes[dt_id][3] += bboxes[dt_id][1]
                idxs[dt_id] = dt['category_id']
                scores[dt_id] = dt['score_3d']
            pick_indexes = batched_nms(bboxes, scores, idxs, iou_threshold=iou2d_threshold)
            if len(pick_indexes) != n_obj:
                predictions[seq_name][frame_name] = [objs[pick_index] for pick_index in pick_indexes]
    dir, _ = os.path.split(pred_path)
    file_name = 'bbox3d_predictions'
    save_file_name = f'{file_name}_2dnms_{iou2d_threshold}.json'
    save_path = os.path.join(dir, save_file_name)
    with open(save_path, 'w') as out_file:
        json.dump(predictions, out_file, indent=2)


def nms_filter_smoke(pred_path):
    with open(pred_path, 'rb') as f:
        predictions = pickle.load(f)
    for image_id, prediction in predictions.items():
        objs = prediction.tolist()
        # each row: [quat with real part first, x, y, z, length, width, height, score, original_index]
        objects = []
        for i, obj in enumerate(objs):
            bbox3d = [*obj[11:15], *obj[8:11], obj[5], obj[7], obj[6], obj[18]]
            objects.append([*bbox3d, float(i)])
        objects = torch.tensor(objects) 
        pick_indexes = NMS3D_cruw(objects, iou_threshold).flatten().to(torch.int).tolist()
        predictions[image_id] = torch.tensor([objs[pick_index] for pick_index in pick_indexes])
    dir, file_name = os.path.split(pred_path)
    file_name = file_name.split('.')[0]
    save_file_name = f'{file_name}_3dnms_{iou_threshold}.pkl'
    save_path = os.path.join(dir, save_file_name)
    with open(save_path, 'wb') as out_file:
        pickle.dump(predictions, out_file)

if __name__ == '__main__':
    iou_threshold = 0.
    iou2d_threshold = 0.75
    # Kitti format
    # prediction_dir = '/data/test_smaple_apollo/prediction/'
    # save_dir = '/data/test_smaple_apollo/3dnms_filtered'
    # nms_filter(prediction_dir, save_dir)

    enable_filter = True
    # SMOKE CRUW3D
    prediction_path = f'/home/andy/ipl/SMOKE/logs/cruw3d/inference/cruw_test/predictions_train.pkl'
    nms_filter_smoke(prediction_path)


    # CRUW format

    '''
    prediction_path = f'outputs/2023-10-25/22-39-08/inference/final/unimonocam/bbox3d_predictions.json' # change
    prediction_format = 'raw' # raw # change
    enable_filter = True # change
    nms_filter_cruw(prediction_path, prediction_format)
    '''
    # nms2d_filter_cruw(prediction_path, prediction_format)

    # for i in range(1, 4):
    #     prediction_path = f'/mnt/disk1/neurlps_exp/cruw_{i}/bbox3d_predictions.json'
    #     prediction_format = 'aggregate' # raw
    #     enable_filter = True
    #     nms_filter_cruw(prediction_path, prediction_format)

    # for i in range(1, 4, 2):
    #     prediction_path = f'/mnt/disk1/neurlps_exp/cruw_{i}_v2/bbox3d_predictions.json'
    #     prediction_format = 'aggregate' # raw
    #     enable_filter = True
    #     nms_filter_cruw(prediction_path, prediction_format)
    
    # prediction_path = f'/mnt/disk1/neurlps_exp/smoke_1/bbox3d_predictions_3dnms_0.0.json'

    # image_roots = ['day', 'day_enh', 'day_night', 'day_night_enh']
    # for img_root in image_roots:
    #     prediction_path = f'/mnt/nas/nas_cruw/neurips_exp/cam_detection_result/dd3d/v2_99/{img_root}/bbox3d_predictions_3dnms_0.0.json'
    #     if not os.path.exists(prediction_path):
    #         continue
    #     prediction_format = 'aggregate' # raw
    #     nms2d_filter_cruw(prediction_path, prediction_format)
