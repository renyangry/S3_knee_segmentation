import os
import mcubes
from pyparsing import OnlyOnce
from mesh_ssm_utils import *
import numpy as np


root_dir = '/home/rgu/Documents'
moving_train_dir = os.path.join(root_dir,'wrapped_label', 'left_femur','train')
original_train_dir = os.path.join(root_dir, 'UK dataset', 'nnUNet_raw', 'Dataset1000_NMDID', 'labelsTr') 
out_path = os.path.join(root_dir, 'mesh_ssm')
json_path = os.path.join(out_path, 'mesh_training.json')
os.makedirs(out_path, exist_ok=True)
fasy_quadric_bin = '/home/rgu/Downloads/Fast-Quadric-Mesh-Simplification-master/bin.Linux'
# @OnlyOnce
# save_training_json(moving_train_dir, original_train_dir, json_path)

with open(json_path, 'r') as f:
    paths = json.load(f)["ssm_list"]
    for filename in paths:
        im = ants.image_read(filename)
        # affine = ants.get_affine(im)
        femur_seg = extract_femur(im, 1)
        # ants.image_write(femur_seg,os.path.join(out_path,os.path.basename(filename)))
        # print(femur_seg.direction) 
        axis = np.where(femur_seg.direction == -1)[0]
        femur_seg.set_origin((0, 0, 0))
        femur_seg_np = femur_seg.numpy()
        lf_seg = np.flip(femur_seg_np, axis)
        lf_smooth = mcubes.smooth(lf_seg, method='gaussian')
        convert_binary2mesh(lf_smooth, filename, os.path.join(out_path, 'mesh'))
        print(f'{filename} done')

        
