#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 12:26:43 2023

@author: renne
"""
import os
import glob
import numpy as np
import nibabel as nib
import nrrd
from scipy import ndimage as ndi
from utils import *

################################# main ######################################
data_root = '/home/rgu/Documents/'
## mac user '/Users/renne/Documents/Imperial/'
which_dataset = 'UK dataset' 
output_path_full = os.path.join(data_root,which_dataset,'leg_img')
check_path_exist(output_path_full)
output_path_side = os.path.join(data_root,which_dataset,'seperated_leg_img')
check_path_exist(output_path_side)


image_path = sorted(glob.glob(os.path.join(data_root,which_dataset,'*.nii.gz')))

for idx in range(len(image_path)):
    img = nib.load(image_path[idx]).get_fdata()
    threshold_mask = global_threshold_mask(img, -400, img.max())
    right_seg, left_seg = split_img(threshold_mask)
    right_body_mask, left_body_mask = generate_body_mask(right_seg), generate_body_mask(left_seg)
    
    right_img, left_img = split_img(img)
    right_leg, left_leg = remove_ct_couch(right_img, right_body_mask), remove_ct_couch(left_img, left_body_mask)
    
    full_leg = reconstruct_img(right_leg, left_leg)
    
    nii_file_writer(image_path[idx],right_leg,'right',output_path_side)
    nii_file_writer(image_path[idx],left_leg,'left',output_path_side)
    nii_file_writer(image_path[idx],full_leg,'leg',output_path_full)


