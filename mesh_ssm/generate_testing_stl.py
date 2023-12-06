import os
import ants
import mcubes
import json
from mesh_ssm_utils import *
import numpy as np


root_dir = '/home/rgu/Documents'
moving_train_dir = os.path.join(root_dir,'wrapped_label', 'left_femur','train')
original_train_dir = os.path.join(root_dir, 'UK dataset', 'nnUNet_raw', 'Dataset1000_NMDID', 'labelsTr') 
test_dir = os.path.join(root_dir, 'test','LF_intermediate')
out_path = os.path.join(root_dir, 'mesh_ssm')
json_path = os.path.join(out_path, 'mesh_training.json')
os.makedirs(out_path, exist_ok=True)


with open(json_path, 'r') as f:
    paths = json.load(f)["testing_sample"] 
    for filename in paths:
        im = ants.image_read(filename)
        femur_seg = ants.utils.pad_image(im, pad_width=[(20, 20), (20, 20), (20, 20)])
        # ants.image_write(femur_seg,os.path.join(out_path,os.path.basename(filename)))
        # reorient femur to align with the new axis with origin at [0,0,0] 
        axis = np.where(femur_seg.direction == -1)[0]
        femur_seg.set_origin((0, 0, 0))
        femur_seg_np = femur_seg.numpy()
        lf_seg = np.flip(femur_seg_np, axis)
        # smooth the femur
        lf_smooth = mcubes.smooth(lf_seg, method='gaussian')
        convert_binary2mesh(lf_smooth, filename, os.path.join(out_path, 'mesh_test'))
        print(f'{filename} done')