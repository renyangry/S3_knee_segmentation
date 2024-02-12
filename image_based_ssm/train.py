
from ssm_model import RecSSM
from ssm_utils import *
from ssm_config import *
import ssm_train
import ssm_test
import ants
import json

# OnlyOnce
check_path_exist(OUT_DIR)
save_training_json(TRAIN_DIR, JSON_PATH)
# generate_fixed_image(ROOT_DIR)

bone = 'left_femur'
# for bone in BONE_STRUCTURE:
print(f'Processing {bone}...')
warped_train_dir = WARPED_DIRS[bone]['train']
results_dir = RESULTS_DIRS[bone]

check_path_exist(warped_train_dir)
check_path_exist(results_dir)

fixed_anatomy = ants.image_read(os.path.join(ROOT_DIR, bone + '.nii.gz'))
#
fixed_surface = generate_surface_img(fixed_anatomy.numpy())
fixed_surface_img = transform_numpy2ants(fixed_anatomy, fixed_surface)
# ants.image_write(fixed_surface_img, os.path.join(ROOT_DIR, bone + '_surface.nii.gz'))
#

with open(JSON_PATH, 'r') as f:
    training_dict = json.load(f)

    if 'left' in bone:
        leg_list = training_dict['left_leg_list']
    else:
        leg_list = training_dict['right_leg_list']

# model = RecSSM(len(leg_list), fixed_anatomy.shape)
# ssm_train.train_ssm(bone, leg_list, fixed_anatomy, model, warped_train_dir, results_dir)
#
model = RecSSM(len(leg_list), fixed_surface_img.shape)
ssm_train.train_ssm(bone, leg_list, fixed_surface_img, model, warped_train_dir, results_dir)
#
del model
gc.collect()
    
    