import os
import glob
import ssm_test
import ants
import json
import mcubes
import numpy as np
from ssm_model import RecSSM
from ssm_utils import *
from ssm_config import *
import ssm_train
from mesh_ssm.mesh_ssm_utils import *


# step 1: generate json sorting all pt files for SSM training
output_dir = '/home/rgu/Documents/new_ssm'
os.makedirs(output_dir, exist_ok=True)
new_json_path = os.path.join(output_dir, 'training.json')
# save_training_json(TRAIN_DIR, new_json_path)


# step 2: mesh conversion first
# with open(new_json_path, 'r') as f:
#     paths = json.load(f)["left_leg_list"]
#     for filename in paths:
#         im = ants.image_read(filename)
#         femur_seg = extract_femur(im, 1)
#         ## ants.image_write(femur_seg, os.path.join(output_dir, os.path.basename(filename)))
#         # reorient femur to align with the new axis with origin at [0,0,0]
#         axis = np.where(femur_seg.direction == -1)[0]
#         femur_seg.set_origin((0, 0, 0))
#         femur_seg_np = femur_seg.numpy()
#         lf_seg = np.flip(femur_seg_np, axis)
#         # smooth the femur
#         lf_smooth = mcubes.smooth(lf_seg, method='gaussian')
#         convert_binary2mesh(lf_smooth, filename, os.path.join(output_dir, 'mesh'))
#         print(f'{filename} done')


# step 3: simplifying mesh
mesh_dir = glob.glob(os.path.join(output_dir, 'mesh', '*.stl'))
sim_mesh_dir = os.path.join(output_dir, 'simplify_mesh')
os.makedirs(sim_mesh_dir, exist_ok=True)
bin_path = '/home/rgu/Downloads/Fast-Quadric-Mesh-Simplification-master/bin.Linux/simplify'
simplify_mesh(os.path.join(output_dir, 'mesh'), sim_mesh_dir, bin_path, 0.4)
rename_mesh4ssm(sim_mesh_dir, os.path.join(output_dir, 'simplify_stl'))
# Shuqiao's method
# for meshes in sorted(mesh_dir):
#     f_out = os.path.join(sim_mesh_dir, os.path.basename(meshes))
#     add_surface_remesh(meshes, clusters=20000, subdivide=3, f_out=f_out, image_plot=False)


# step 4: iSSM image registration + training
bone = 'left_femur'
print(f'Processing {bone}...')
warped_train_dir = WARPED_DIRS[bone]['train']
results_dir = RESULTS_DIRS[bone]
os.makedirs(warped_train_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

fixed_anatomy = ants.image_read(os.path.join(ROOT_DIR, bone + '.nii.gz'))

with open(new_json_path, 'r') as f:
    training_dict = json.load(f)
    if 'left' in bone:
        leg_list = training_dict['left_leg_list']
    else:
        leg_list = training_dict['right_leg_list']

model = RecSSM(len(leg_list), fixed_anatomy.shape)
ssm_train.train_ssm(bone, leg_list, fixed_anatomy, model, warped_train_dir, results_dir)
del model
gc.collect()