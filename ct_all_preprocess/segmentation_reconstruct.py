#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 22:48:14 2023

@author: renne
"""

import os
import glob
from typing import Optional

import numpy as np
# from scipy import stats
import nibabel as nib
from scipy.ndimage import label
from utils import check_path_exist, nii_file_writer, split_img


def bone_seg_separation(img, seg_value):
    mask_img = np.zeros_like(img)
    if seg_value is None:
        return mask_img
    else:
        # seg_value = np.array(seg_value)
        # valid_value = seg_value[seg_value>0]
        # for value in valid_value:
        mask_img[seg_value == img] = 1
        return mask_img


# path define
data_root = '/home/rgu/Documents/'
which_dataset = 'UK dataset'
output_path = os.path.join(data_root, which_dataset, 'seperated_leg_seg')
check_path_exist(output_path)

seg_img = sorted(glob.glob(os.path.join(data_root, which_dataset, 'seg_img', '*.nii.gz')))
left_femur_threshold = [2178, 1765, 2178, None, None, None, None, None, None, None, None, None, None]
left_tibia_threshold = [1410, None, 1410, None, None, None, None, None, None, None, None, None, 2254]
right_femur_threshold = [None, 1500, 1874, 1943, 1874, 1874, 1943, None, 1942, 1874, 1874, 1943, 1943]
right_tibia_threshold = [2478, None, 2478, None, None, None, None, None, None, None, None, None, None]

for idx in range(len(seg_img)):
    seg = nib.load(seg_img[idx]).get_fdata()
    right_seg, left_seg = split_img(seg)

    femur_left = bone_seg_separation(left_seg, left_femur_threshold[idx])
    left_labelled_volume, left_num_features = label(femur_left)
    print(f"Number of connected components of left femur: {left_num_features}")
    if left_num_features > 0:
        nii_file_writer(seg_img[idx], femur_left, 'seg_left', output_path)

    femur_right = bone_seg_separation(right_seg, right_femur_threshold[idx])
    right_labelled_volume, right_num_features = label(femur_right)
    if right_num_features > 1:
        sizes = np.bincount(right_labelled_volume.ravel())
        bone_sizes: Optional[bool] = (sizes > 1000)
        bone_sizes[0] = 0
        femur_right = bone_sizes[right_labelled_volume]
    right_labelled_volume1, right_num_features1 = label(femur_right)
    print(f"Number of connected components of right femur: {right_num_features1}")
    if right_num_features > 0:
        nii_file_writer(seg_img[idx], femur_right, 'seg_right', output_path)

    tibia_left = bone_seg_separation(left_seg, left_tibia_threshold[idx])
    labelled_volume, num_features = label(tibia_left)
    if num_features > 1:
        sizes = np.bincount(labelled_volume.ravel())
        bone_sizes: Optional[bool] = (sizes > 1000)
        bone_sizes[0] = 0
        tibia_left = bone_sizes[labelled_volume]
    labelled_volume1, num_features1 = label(tibia_left)
    print(f"Number of connected components of left tibia: {right_num_features1}")
    # nii_file_writer(seg_img[idx], tibia_left, 'left_tibia', output_path)

    tibia_right = bone_seg_separation(right_seg, right_tibia_threshold[idx])

    if (left_femur_threshold[idx] is not None) or (left_tibia_threshold[idx] is not None):
        nii_file_writer(seg_img[idx], femur_left + tibia_left, 'left', output_path)
    if (right_femur_threshold[idx] is not None) or (right_tibia_threshold[idx] is not None):
        nii_file_writer(seg_img[idx], femur_right + tibia_right, 'right', output_path)
