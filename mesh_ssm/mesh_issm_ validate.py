import os
import ants
import mcubes
import json
from mesh_ssm_utils import *
import numpy as np


root_dir = '/home/rgu/Documents'
condition = ['proximal_only', 'add_distal', 'reduced_distal']
moving_train_dir = [os.path.join(root_dir, 'ssm_results', 'left_femur', condition[i]) for i in range(len(condition))]
raw_test_dir = os.path.join(root_dir, 'test', 'left_femur')
test_dir = os.path.join(root_dir, 'test', 'left_femur_intermediate')
out_path = os.path.join(root_dir, 'new_ssm')
json_path = os.path.join(out_path, 'issm_validation.json')

train_list = []
test_list = []
for filename in os.listdir(raw_test_dir):
    train_list.append(os.path.join(test_dir, filename))
    test_list.extend(os.path.join(moving_train_dir[i], filename) for i in range(len(moving_train_dir)))

with open(json_path, 'w') as f:
    json.dump({'cropped_manual_seg': train_list, 'reconstructed_seg': test_list}, f)



with open(json_path, 'r') as f:
    paths = json.load(f)["reconstructed_seg"] 
    for filename in paths:
        im = ants.image_read(filename)
        femur_seg = ants.utils.pad_image(im, pad_width=[(20, 20), (20, 20), (20, 20)])
        axis = np.where(femur_seg.direction == -1)[0]
        femur_seg.set_origin((0, 0, 0))
        femur_seg_np = femur_seg.numpy()
        lf_seg = np.flip(femur_seg_np, axis)
        # smooth the femur
        lf_smooth = mcubes.smooth(lf_seg, method='gaussian')
        cond = filename.split('/')[-2]
        convert_binary2mesh(lf_smooth, filename, os.path.join(out_path, 'mesh_issm_validation',cond))
        print(f'{filename} done')


for i in range(len(condition)):
    mesh_dir = glob.glob(os.path.join(out_path, 'mesh_issm_validation', condition[i], '*.obj'))
    sim_mesh_dir = os.path.join(out_path, 'simplify_mesh_issm_validation', condition[i])
    os.makedirs(sim_mesh_dir, exist_ok=True)
    bin_path = '/home/rgu/Downloads/Fast-Quadric-Mesh-Simplification-master/bin.Linux/simplify'
    simplify_mesh(os.path.join(out_path, 'mesh_issm_validation',condition[i]), sim_mesh_dir, bin_path, 0.4)
    rename_mesh4ssm(sim_mesh_dir, os.path.join(out_path, 'simplify_issm_validation_stl', condition[i]))

os.system('rm -r ' + os.path.join(out_path, 'simplify_mesh_issm_validation'))