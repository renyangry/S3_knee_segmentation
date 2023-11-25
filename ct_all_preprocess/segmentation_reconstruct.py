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
import nibabel as nib
from scipy.ndimage import label
from utils import check_path_exist, nii_file_writer, split_img, bone_seg_separation, clean_up_seg, single_bone_seg_process





# path define
data_root = '/home/rgu/Documents/'
which_dataset = 'UK dataset'
output_path = os.path.join(data_root, which_dataset, 'seperated_leg_seg_db')
check_path_exist(output_path)

seg_img = sorted(glob.glob(os.path.join(data_root, which_dataset, 'seg_img', '*.nii.gz')))
left_femur_threshold = [2178, 1765, 2178, None, 1140, None, 2178, 2178, 648, None, 2178, 2178, 2254, 2254, None, None, 2254, None, None, None, None, None, 2178, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 1411, None, 1943, None, None, None, None, None, None, None, None]
left_tibia_threshold = [1410, None, 1410, None, None, None, 1210, 1410, None, None, 2478, None, 1468, None, None, None, 2561, 1409, None, None, 1410, None, None, None, None, None, None, None, None, 5524, None, None, None, 2254, None, 2254, None, 2254, 1667, None, None, None, None, None, 2254, None, 1311, None, None]
right_femur_threshold = [None, 1500, 1874, 1943, 934, 1874, None, 1874, 968, 1874, 1874, None, 1943, 1943, 1874, 1874, 1943, 1874, 1942, 1316, 1874, 1943, 1874, 1515, 2604, 1874, 1943, None, 1942, 1943, 1874, 1874, 1943, 1943, 1943, 1943, 1943, 1943, None, 1943, None, 1874, 1228, 2519, 2604, 1599, 1560, 2604, 2519]
right_tibia_threshold = [2478, None, 2478, None, 1342, None, 1409, 2478, 511, None, 1409, None, None, 2561, 2478, 2478, 1468, None, 2560, 1804, 2478, 2561, 2478, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]

 
assert len(left_femur_threshold) == len(left_tibia_threshold)
assert len(right_femur_threshold) == len(right_tibia_threshold)
assert len(left_femur_threshold) == len(seg_img)
assert len(right_femur_threshold) == len(seg_img)


left_index = []
right_index = []
for idx in range(len(left_femur_threshold)):
    if (left_femur_threshold[idx] is not None) and (left_tibia_threshold[idx] is not None):
        left_index.append(idx)
    if (right_femur_threshold[idx] is not None) and (right_tibia_threshold[idx] is not None):
        right_index.append(idx)

# print(left_index)
# print(right_index)

left_femur_index = [idx for idx in range(len(left_femur_threshold)) if {left_femur_threshold[idx] is not None} & {idx not in left_index}]
left_tibia_index = [idx for idx in range(len(left_tibia_threshold)) if {left_tibia_threshold[idx] is not None} & {idx not in left_index}]
right_femur_index = [idx for idx in range(len(right_femur_threshold)) if {right_femur_threshold[idx] is not None} & {idx not in right_index}]
right_tibia_index = [idx for idx in range(len(right_tibia_threshold)) if {right_tibia_threshold[idx] is not None} & {idx not in right_index}]

# print(left_femur_index)
# print(left_tibia_index)
# print(right_femur_index)
# print(right_tibia_index)


for idx in left_index:
    seg = nib.load(seg_img[idx]).get_fdata()
    right_seg, left_seg = split_img(seg)
    femur_left = bone_seg_separation(left_seg, left_femur_threshold[idx])
    tibia_left = bone_seg_separation(left_seg, left_tibia_threshold[idx])
    cleaned_femur_left = clean_up_seg(femur_left, 1000)
    cleaned_tibia_left = clean_up_seg(tibia_left, 1000) * 2
    left_leg = cleaned_femur_left + cleaned_tibia_left
    nii_file_writer(seg_img[idx], left_leg, 'left', output_path)
    print(f"Case {idx+1} done")
print('left leg done')

for idx in right_index:
    seg = nib.load(seg_img[idx]).get_fdata()
    right_seg, left_seg = split_img(seg)
    femur_right = bone_seg_separation(right_seg, right_femur_threshold[idx])
    tibia_right = bone_seg_separation(right_seg, right_tibia_threshold[idx])
    cleaned_femur_right = clean_up_seg(femur_right, 1000)
    cleaned_tibia_right = clean_up_seg(tibia_right, 1000) * 2
    right_leg = cleaned_femur_right + cleaned_tibia_right
    nii_file_writer(seg_img[idx], right_leg, 'right', output_path)
    print(f"Case {idx+1} done")
print('right leg done')

single_bone_seg_process(seg_img, left_femur_threshold, left_femur_index, output_path, 'femur_left')
print('left femur done')
single_bone_seg_process(seg_img, left_tibia_threshold, left_tibia_index, output_path, 'tibia_left')
print('left tibia done')
single_bone_seg_process(seg_img, right_femur_threshold, right_femur_index, output_path, 'femur_right')
print('right femur done')
single_bone_seg_process(seg_img, right_tibia_threshold, right_tibia_index, output_path, 'tibia_right')
print('right tibia done')

