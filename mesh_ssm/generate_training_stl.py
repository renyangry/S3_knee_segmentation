import os

from pyparsing import OnlyOnce
from mesh_ssm_utils import *
import numpy as np


root_dir = '/home/rgu/Documents'
moving_train_dir = os.path.join(root_dir,'wrapped_label', 'left_femur','train')
original_train_dir = os.path.join(root_dir, 'UK dataset', 'nnUNet_raw', 'Dataset1000_NMDID', 'labelsTr') 
out_path = os.path.join(root_dir, 'mesh_ssm')
json_path = os.path.join(out_path, 'mesh_training.json')
os.makedirs(out_path, exist_ok=True)

# @OnlyOnce
# save_training_json(moving_train_dir, original_train_dir, json_path)

# interpolate_all_masks(out_path, os.path.join(out_path,'tmp'))
with open(json_path, 'r') as f:
    paths = json.load(f)["ssm_list"]
    for filename in paths:
        im = ants.image_read(filename)
        # affine = ants.get_affine(im)
        femur_seg = extract_femur(im, 0)
        femur_seg = femur_seg.numpy()
        femur_mesh = convert_binary2mesh(femur_seg, filename, os.path.join(out_path,'mesh'))
        
