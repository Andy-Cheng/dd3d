import numpy as np
import matplotlib.pyplot as plt

viz_npz_file = '/home/andy/Downloads/tmp_dd3d/visualization.npz'

viz_npz = np.load(viz_npz_file)

for key in viz_npz.keys():
    key_split = key.split('/')
    name = '_'.join(key_split)
    plt.imsave(f'/home/andy/Downloads/tmp_dd3d/tmp_viz/{name}.png', viz_npz[key])