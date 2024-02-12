# utils.py

import json
import os
import re
import ants
import numpy as np
import gc
import SimpleITK as sitk

# def compute_min_max(label, execpt_axis):
#     indices = np.nonzero(label.sum(axis=execpt_axis) > 0)[0]
#     axis_min = max(0, indices.min() - 10)
#     axis_max = min(label.shape[0], indices.max() + 10)
#     return axis_min, axis_max

def extract_one_bone(full_leg_seg, label):
    # x_min, x_max = compute_min_max(mask, (1, 2))
    # y_min, y_max = compute_min_max(mask, (0, 2))
    # z_min, z_max = compute_min_max(mask, (0, 1))
    # cropped_new_arr = mask[x_min:x_max, y_min:y_max, z_min:z_max]
    mask_img = ants.utils.get_mask(full_leg_seg, label, label+1)
    fixed_cropped = ants.utils.crop_image(mask_img, mask_img)
    cropped_padded = ants.utils.pad_image(fixed_cropped, pad_width=[(20, 20), (20, 20), (20, 20)])
    return cropped_padded


def generate_fixed_image(ROOT_DIR):
    #  because two dataset is mixed, the fixed image is calculated by averaging N images from two datasets
    # fixed_list = random.sample(left_leg_list_uk, 3) + random.sample(left_leg_list_us, 3)
    # fixed_img = ants.average_images(fixed_list)
    print('generating fixed image...')
    
    left_fixed_img = ants.image_read('/home/rgu/Documents/UK dataset/nnUNet_raw/Dataset1000_NMDID/labelsTr/halvleg_1064.nii.gz')
    fixed_left_femur = extract_one_bone(left_fixed_img, 1)
    ants.image_write(fixed_left_femur, os.path.join(ROOT_DIR, 'left_femur.nii.gz'))
    print('left femur done')
    left_fixed_img1 = ants.image_read('/home/rgu/Documents/UK dataset/nnUNet_raw/Dataset1000_NMDID/labelsTr/halvleg_1007.nii.gz')
    fixed_left_tibia = extract_one_bone(left_fixed_img1, 2)
    ants.image_write(fixed_left_tibia, os.path.join(ROOT_DIR, 'left_tibia.nii.gz'))
    print('left tibia done')
    
    right_fixed_img = ants.image_read('/home/rgu/Documents/UK dataset/nnUNet_raw/Dataset1000_NMDID/labelsTr/halvleg_062.nii.gz')
    fixed_right_femur = extract_one_bone(right_fixed_img, 1)
    ants.image_write(fixed_right_femur, os.path.join(ROOT_DIR, 'right_femur.nii.gz'))
    print('right femur done')
    fixed_right_tibia = extract_one_bone(right_fixed_img, 2)
    ants.image_write(fixed_right_tibia, os.path.join(ROOT_DIR, 'right_tibia.nii.gz'))
    print('right tibia done')
    del fixed_left_femur, fixed_left_tibia, fixed_right_femur, fixed_right_tibia
    gc.collect()
        

def OnlyOnce(func):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return func(*args, **kwargs)
    wrapper.has_run = False
    return wrapper


@OnlyOnce
def save_training_json(moving_train_dir, json_path):
    left_leg_list_uk = []
    left_leg_list_us = []

    pattern_4_digits = r'halvleg_(\d{4})\.nii\.gz'
    pattern_6_digits_starting_with_2 = r'halvleg_(2\d{5})\.nii\.gz'
    for filename in os.listdir(moving_train_dir):
        if re.match(pattern_4_digits, filename):
            left_leg_list_uk.append(os.path.join(moving_train_dir, filename))
        elif re.match(pattern_6_digits_starting_with_2, filename):
            left_leg_list_us.append(os.path.join(moving_train_dir, filename))
    
    left_leg_list = left_leg_list_uk + left_leg_list_us
    all_files = set(os.listdir(moving_train_dir))
    left_set = set([os.path.basename(f) for f in left_leg_list])
    right_leg_set = all_files - left_set
    right_leg_list = [os.path.join(moving_train_dir, f) for f in right_leg_set]
 
    with open(json_path, 'w') as f:
        json.dump({'left_leg_list': left_leg_list, 'right_leg_list': right_leg_list}, f)
	

def check_path_exist(path_):
    os.makedirs(path_, exist_ok=True)

def compute_efficiency(start_time, end_time):
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    return minutes, seconds

def generate_surface_img(nifti_numpy):
    ref_image = sitk.GetImageFromArray(nifti_numpy)
    ref_image_int = sitk.Cast(ref_image, sitk.sitkUInt8)
    ref_surface = sitk.LabelContour(ref_image_int)
    ref_surface_array = sitk.GetArrayFromImage(ref_surface)
    return ref_surface_array

def transform_numpy2ants(original_ants, surface_array):
    spacing = original_ants.spacing
    origin = original_ants.origin
    direction = original_ants.direction
    surface_image = ants.from_numpy(surface_array)
    surface_image.set_spacing(spacing)
    surface_image.set_origin(origin)
    surface_image.set_direction(direction)
    return surface_image