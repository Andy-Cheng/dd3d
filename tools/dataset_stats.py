import json
import numpy as np

def calc_class_stats(train_data):
    class_stats = {}
    for sample in train_data:
        for obj in sample['objs']:
            if not obj['obj_type'] in class_stats:
                obj_stats = {'obj_count': 0, 'lwh': np.array([0, 0, 0], dtype=np.float32)}
                class_stats[obj['obj_type']] = obj_stats
            class_stats[obj['obj_type']]['obj_count'] += 1
            class_stats[obj['obj_type']]['lwh'] += np.array(obj['scale'], dtype=np.float32)
    return class_stats

if  __name__ == '__main__':
    with open('dataset_root/kradar_label/kradar_cam_aligned_v3.json', 'r') as f:
        train_data = json.load(f)['train']
    
    class_stats = calc_class_stats(train_data)
    for obj_type in class_stats:
        class_stats[obj_type]['lwh'] /= class_stats[obj_type]['obj_count']
        class_stats[obj_type]['lwh'] = class_stats[obj_type]['lwh'].tolist()
    with open('/mnt/nas_kradar/kradar_dataset/class_stats.json', 'w') as out_f:
        json.dump(class_stats, out_f, indent=2)

