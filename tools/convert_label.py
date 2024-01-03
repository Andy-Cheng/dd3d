from math import pi
import os
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import json
from math import ceil
import torch
from collections import defaultdict

# 3D util
def radx_to_matrix(rotx, N):
    device = rotx.device

    cos, sin = rotx.cos(), rotx.sin()

    i_temp = torch.tensor([[1, 0, 0],
                            [0, 1, -1],
                            [0, 1, 1]]).to(dtype=torch.float32,
                                            device=device)
    rx = i_temp.repeat(N, 1).view(N, -1, 3)

    rx[:, 1, 1] *= cos
    rx[:, 1, 2] *= sin
    rx[:, 2, 1] *= sin
    rx[:, 2, 2] *= cos

    return rx

def rady_to_matrix(rotys, N):
    device = rotys.device

    cos, sin = rotys.cos(), rotys.sin()

    i_temp = torch.tensor([[1, 0, 1],
                            [0, 1, 0],
                            [-1, 0, 1]]).to(dtype=torch.float32,
                                            device=device)
    ry = i_temp.repeat(N, 1).view(N, -1, 3)

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
    rz = i_temp.repeat(N, 1).view(N, -1, 3)

    rz[:, 0, 0] *= cos
    rz[:, 0, 1] *= sin
    rz[:, 1, 0] *= sin
    rz[:, 1, 1] *= cos

    return rz

def encode_box3d(eulers, dims, locs):
    '''
    construct 3d bounding box for each object.
    Args:
        rotys: rotation in shape N
        dims: dimensions of objects
        locs: locations of objects

    Returns:
    '''
    N = eulers.shape[0]
    rx = radx_to_matrix(eulers[:, 0], N) # (N, 3, 3)
    ry = rady_to_matrix(eulers[:, 1], N) # (N, 3, 3)
    rz = radz_to_matrix(eulers[:, 2], N) # (N, 3, 3)

    dims = dims.view(-1, 1).repeat(1, 8) # [[eight l], [eight w], [eight h]]  # (N*3, 8)
    dims[::3, :4], dims[2::3, :4] = 0.5 * dims[::3, :4], 0.5 * dims[2::3, :4]
    dims[::3, 4:], dims[2::3, 4:] = -0.5 * dims[::3, 4:], -0.5 * dims[2::3, 4:]
    dims[1::3, :4], dims[1::3, 4:] = 0.5 * dims[1::3, 4:], -0.5 * dims[1::3, 4:]
    index = torch.tensor([[4, 0, 1, 2, 3, 5, 6, 7],
                            [4, 5, 0, 1, 6, 7, 2, 3],
                            [4, 5, 6, 0, 1, 2, 3, 7]]).repeat(N, 1)
    box_3d_object = torch.gather(dims, 1, index)
    box_3d =  box_3d_object.view(N, 3, -1)
    box_3d = torch.matmul(torch.matmul(rz, torch.matmul(rx,ry)) , box_3d_object.view(N, 3, -1) )
    box_3d += locs.unsqueeze(-1)

    return box_3d

def quaternion_upper_hemispher(q):
    """
    The quaternion q and −q represent the same rotation be-
    cause a rotation of θ in the direction v is equivalent to a
    rotation of 2π − θ in the direction −v. One way to force
    uniqueness of rotations is to require staying in the “upper
    half” of S 3 . For example, require that a ≥ 0, as long as
    the boundary case of a = 0 is handled properly because of
    antipodal points at the equator of S 3 . If a = 0, then require
    that b ≥ 0. However, if a = b = 0, then require that c ≥ 0
    because points such as (0,0,−1,0) and (0,0,1,0) are the
    same rotation. Finally, if a = b = c = 0, then only d = 1 is
    allowed.
    :param q:
    :return:
    """
    a, b, c, d = q
    if a < 0:
        q = -q
    if a == 0:
        if b < 0:
            q = -q
        if b == 0:
            if c < 0:
                q = -q
            if c == 0:
                q[3] = 1

    return q

def conver_label(Tc_l, rot, t, scale):
    Tl_o = np.identity(4)
    Tl_o[:3, :3] = Rotation.from_euler('zyx', [rot['z'], rot['y'], rot['x']]).as_matrix()
    Tl_o[:3, 3] = np.array([t['x'], t['y'], t['z']])
    Tc_o = np.matmul(Tc_l, Tl_o)
    R = Tc_o[:3, :3]
    T = Tc_o[:3, 3]
    r = Rotation.from_matrix(R)
    yaw, pitch, roll = r.as_euler('yxz')
    tx, ty, tz = T
    quat = quaternion_upper_hemispher(r.as_quat()).tolist() # scalar last (x, y, z, w)

    return [pitch, yaw, roll], [tx, ty, tz], [scale['x'], scale['y'], scale['z']], quat

# return bool indicating in fov and left, right,  width, height of projected 2d bbox
def in_fov(euler, position, scale, intrinsics, img_w=1280., img_h=720.):
    # First, check if the object is behind the cameara
    if position[2] < 0.1:
        return False, 0, 0, 0, 0
    box3d = encode_box3d(torch.tensor(euler).view(-1, 3), torch.tensor(scale).view(-1, 3), torch.tensor(position).view(-1, 3)).numpy()[0]
    box3d_homo = np.ones((4, 8), dtype=box3d.dtype)
    box3d_homo[:3, :] = box3d
    img_cor_points = np.dot(intrinsics, box3d_homo)
    img_cor_points = img_cor_points.T
    img_cor_points[:, 0] /= img_cor_points[:, 2]
    img_cor_points[:, 1] /= img_cor_points[:, 2]
    l, t, r, b = img_cor_points[:, 0].min(), img_cor_points[:, 1].min(), img_cor_points[:, 0].max(), img_cor_points[:, 1].max()
    return (img_cor_points.min() >= 0.) and (img_cor_points[:, 0].max() < img_w) and (img_cor_points[:, 1].max() < img_h), l.item(), t.item(), (r - l).item(), (b - t).item()
    

def normalize_angle(angle):
    while angle < 0:
        angle += (2 * pi)
    angle %= (2 * pi)
    if angle > pi:
        angle -= (2 * pi)
    return angle
    

def get_target_seq_frames(samples_txt, target_seq, exclude_seq_frame_json):
    seq_frames = defaultdict(list)
    with open(samples_txt, 'r') as sample_txt_file:
        lines = sample_txt_file.readlines()
    for line in lines:
        seq, frame = line.split(',')
        if not(int(seq) in target_seq):
            continue
        frame = frame.split('_')[1].split('.')[0]
        if not(int(frame) in exclude_seq_frame_json[seq]):
            seq_frames[seq].append(frame)
    return  seq_frames


def get_labels(seq_frames, ds_root, calib_relative_path):
    new_labels = []
    for seq, frames in seq_frames.items():
        print(f'Now processing {seq}')
        for frame_name in tqdm(frames):
            label_path = os.path.join(ds_root, seq, 'label', f'{frame_name}.json')
            if not os.path.exists(label_path):
                print(f'{frame_name} has no label')
                continue
            with open(os.path.join(ds_root, seq, calib_relative_path), 'r') as calib_file:
                cam_calib = json.load(calib_file)
            with open(label_path, 'r') as label_file:
                label = json.load(label_file)
            P2 = cam_calib['intrinsic']
            P1 = cam_calib['extrinsic']
            P2 = np.concatenate([np.array(P2, dtype=np.float32).reshape(3, 3), np.zeros((3, 1))], axis=1)
            P1 = np.array(P1, dtype=np.float32).reshape(4, 4)
            objs = []
            for obj in label:
                if obj is None:
                    continue
                psr = obj['psr']
                euler, position, scale, quat = conver_label(P1, psr['rotation'], psr['position'], psr['scale'])
                is_in_fov, l, t, w, h = in_fov(euler, position, scale, P2)
                if not is_in_fov or (np.linalg.norm(np.array(position)) > distance_threshold):
                    continue
                if is_kitti_format:
                    # assume lidar's negative z axis is alignmed with camera's y axis, in SMOKE car's head point to positive x axis but annotation point to z axis (in camera coordinate)thus - pi/2
                    ry = -psr['rotation']['z'] - pi/2
                    converted_obj = {'obj_id': obj['obj_id'], 'obj_type': obj['obj_type'], 'ry': normalize_angle(ry), 'position': position, 'scale': [scale[0], scale[2], scale[1]], '2dbbox': [l, t, w, h]}  # Change to [scale[1], scale[2], scale[0]]
                else:
                    converted_obj = {'obj_id': obj['obj_id'], 'obj_type': obj['obj_type'], 'euler': euler, 'position': position, 'scale': scale, 'quat': quat, '2dbbox': [l, t, w, h]}
                objs.append(converted_obj)
            label_frame = {'seq': seq, 'frame': frame_name, 'objs': objs}
            new_labels.append(label_frame)
    return new_labels


if __name__ == '__main__':
    train_samples = '/home/andy/ipl/CenterPoint/configs/kradar/resources/split/train.txt'
    test_samples = '/home/andy/ipl/CenterPoint/configs/kradar/resources/split/test.txt'
    with open('/mnt/nas_kradar/kradar_dataset/bad_front.json', 'r') as json_file:
        exclude_seq_frame_json = json.load(json_file)
    save_name = 'kradar_cam_aligned_v3_all.json'
    save_root_path = '/home/andy/ipl/dd3d/dataset_root/kradar_label'
    is_kitti_format = False # kitti format means in object coordinate frame, x is forward (L), y is down (H), z is left (W)
    kradar_root = '/home/andy/ipl/dd3d/dataset_root/kradar'
    calib_realtive_path = 'calib/camera/cam-front-undistort.json'
    print(f'Start to preparing files in {save_root_path}')
    # target_seq = [5, 7, 9, 10, 11, 12, 13, 15, 16, 17, 19,  22, 41]
    target_seq = list(range(1, 59))
    for seq_to_remove in [51, 52, 57, 58]:
        target_seq.remove(seq_to_remove)

    distance_threshold = 80 # filter out instances larger than distance (meter)
    train_seq_frames = get_target_seq_frames(train_samples, target_seq, exclude_seq_frame_json)
    print('Start preparing train label.')
    train_labels = get_labels(train_seq_frames, kradar_root, calib_realtive_path)
    print('Finish train label generation.')
    print('Start preparing test label.')
    test_seq_frames = get_target_seq_frames(test_samples, target_seq, exclude_seq_frame_json)
    test_labels = get_labels(test_seq_frames, kradar_root, calib_realtive_path)
    print('Finish test label generation.')
    print(f'Writing to {save_name}')
    output_labels = {} # {split_type: [{obj_type, obj_id, objs: []}]}
    output_labels['train'] = train_labels
    output_labels['test'] = test_labels
    with open(os.path.join(save_root_path, save_name), 'w') as output_labels_file:
        json.dump(output_labels, output_labels_file, indent=2)