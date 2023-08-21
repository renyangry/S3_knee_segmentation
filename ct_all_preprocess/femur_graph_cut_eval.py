import os
import glob
import nrrd
import csv
from typing import List, Any
import numpy as np
import nibabel as nib
from utils import check_num_of_pt_4eval


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def find_start_end_slices_histogram(volume):
    z_histogram = np.sum(volume, axis=(0, 1))  # Calculate histograms along the Z-axis
    nonzero_slices = np.where(z_histogram > 0)[0]
    if len(nonzero_slices) > 0:
        start_slice = nonzero_slices[0]
        end_slice = nonzero_slices[-1]
        return start_slice, end_slice
    else:
        return None, None


data_root = '/home/rgu/Documents/'
which_dataset = 'UK dataset'
gt_path = sorted(glob.glob(os.path.join(data_root, which_dataset, 'seperated_leg_seg', 'seg_*.nii.gz')))
cut_path = sorted(glob.glob(os.path.join(data_root, which_dataset, 'results', 'seg_*.nrrd')))
check_num_of_pt_4eval(gt_path, cut_path)

eval_gt_path = []
cut_file_path = []
for idx, gt_file_path in enumerate(gt_path):
    gt_file_name = os.path.split(gt_file_path)[-1][0:16]
    cut_index = next(
        (i for i, cut_file_path in enumerate(cut_path) if os.path.split(cut_file_path)[-1][0:16] == gt_file_name), None)
    if cut_index is not None:
        eval_gt_path.append(gt_path[idx])
        cut_file_path.append(cut_path[cut_index])

check_num_of_pt_4eval(eval_gt_path, cut_file_path)

DSC = []
for i in range(len(eval_gt_path)):
    gt = nib.load(eval_gt_path[i]).get_fdata()
    automated, header = nrrd.read(cut_file_path[i])
    reorientation_automated = np.flip(automated, axis=(0, 1))
    start_slice, end_slice = find_start_end_slices_histogram(gt)
    dice_score = dice_coef(gt[:, :, start_slice:end_slice], reorientation_automated[:, :, start_slice:end_slice])
    print(f'For A and B, dice coefficient is {dice_score}')
    DSC.append(dice_score)

print('evaluation on graph cut completed')

csv_file_path = os.path.join(data_root, which_dataset, 'results', 'result.csv')
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Image Name', 'DSC'])
    for eval_path, dsc_value in zip(eval_gt_path, DSC):
        image_name = os.path.split(eval_path)[-1][0:16]
        csv_writer.writerow([image_name, dsc_value])

print(f"CSV file '{csv_file_path}' generated successfully.")
