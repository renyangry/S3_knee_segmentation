import os
import glob
import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output


def normalisation(ori_ct,pixel_size):
    ct = resample_to_output(ori_ct,pixel_size)
    sampled_ct = ct.get_fdata()
    sampled_ct -= np.min(sampled_ct)
    normalised_ct = ct/np.max(sampled_ct).astype('float')
    return normalised_ct


data_root = '/home/rgu/Documents/'
which_dataset = 'UK dataset'
left_leg_img = sorted(glob.glob(os.path.join(data_root, which_dataset, 'leg_img', 'left_*.nii.gz')))
right_leg_img = sorted(glob.glob(os.path.join(data_root, which_dataset, 'leg_img', 'right_*.nii.gz')))
for pt in range(len(left_leg_img)):
    assert len(left_leg_img) == len(right_leg_img)
    left_ct = normalisation(left_leg_img[pt])
    right_ct = normalisation(right_leg_img[pt])

