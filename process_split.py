import os
from tqdm import tqdm

split_path = '/home/andy/ipl/dd3d/datasets/kitti_depth/eigen_zhou_files.txt'
new_split_path = '/home/andy/ipl/dd3d/datasets/kitti_depth/filtered_eigen_zhou_files.txt'
kitti_depth_root = '/home/andy/ipl/dd3d/datasets/kitti_depth/train'
if __name__ == '__main__':
    new_split = []
    with open(split_path, 'r') as _f:
        lines = _f.readlines()
        split = [line.split(' ')[0] for line in lines]
    for sample_id in tqdm(split):
        date, folder, sensor, _, file_name = sample_id.split("/")
        file_name = file_name.split(".")[0]
        # Get Depth
        depth_image_name = os.path.join(kitti_depth_root, folder, \
                                                'proj_depth',
                                                'groundtruth',
                                                sensor,
                                                f'{file_name}.png'
                                                )
        if not os.path.exists(depth_image_name):
            print(f'{folder} {sensor} {file_name} not exists')
        else:
            new_split.append(sample_id)
    with open(new_split_path, 'w') as _f:
        for sample_id in new_split:
            _f.write(sample_id+'\n')
    