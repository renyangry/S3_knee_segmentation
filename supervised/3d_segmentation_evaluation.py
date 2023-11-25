import os
import glob
import csv
import numpy as np
import nibabel as nib
from ct_all_preprocess.utils import check_num_of_pt_4eval


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.00001
    return (2. * intersection ) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def other_eval_metrics(y_true: object, y_pred: object) -> object:
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    TP = np.sum(np.logical_and(y_pred_f == 1, y_true_f == 1))
    TN = np.sum(np.logical_and(y_pred_f == 0, y_true_f == 0))
    FP = np.sum(np.logical_and(y_pred_f == 1, y_true_f == 0))
    FN = np.sum(np.logical_and(y_pred_f == 0, y_true_f == 1))
    smooth = 0.00001
    # Calculate Recall
    recall = TP / (TP + FN)
    # Calculate Precision
    precision = TP / (TP + FP)
    # Calculate Dice
    Dice = (2 * TP) / (2 * TP + FP + FN + smooth)
    # Calculate Specificity
    Jaccard = Dice / (2 - Dice)
    # Calculate False Positive Rate
    FPR = FP / (FP + TN)
    # Calculate False Negative Rate
    FNR = FN / (TP + FN)
    # Calculate Percentage of Wrong Classifications
    PWC = 100 * (FN + FP) / (TP + FN + FP + TN)
    return TP, TN, FP, FN, recall, precision, FPR, FNR, PWC, Jaccard, Dice


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
network = 'VNet'
test_folder = 'diceloss'
gt_path = sorted(glob.glob(os.path.join(data_root, which_dataset, 'seperated_leg_seg', 'seg_*.nii.gz')))
seg_path = sorted(glob.glob(os.path.join(data_root, which_dataset, network, test_folder, 'results', '*.nii.gz')))
check_num_of_pt_4eval(gt_path, seg_path)

eval_gt_path = []
seg_file_path = []
for idx, gt_file_path in enumerate(gt_path):
    gt_file_name = os.path.split(gt_file_path)[-1].split('seg_')[1].split('_dicom.nii.gz')[0]
    seg_index = next(
        (i for i, seg_file_path in enumerate(seg_path) if
         os.path.split(seg_file_path)[-1].split('_20')[0] == gt_file_name), None)
    if seg_index is not None:
        eval_gt_path.append(gt_path[idx])
        seg_file_path.append(seg_path[seg_index])

check_num_of_pt_4eval(eval_gt_path, seg_file_path)

DSC = []
statistics = []
for i in range(len(eval_gt_path)):
    gt = nib.load(eval_gt_path[i]).get_fdata()
    automated = nib.load(seg_file_path[i]).get_fdata()
    start_slice, end_slice = find_start_end_slices_histogram(gt)
    dice_score = dice_coef(gt[:, :, start_slice:end_slice], automated[:, :, start_slice:end_slice])
    print(f'For image {i + 1}, dice coefficient is {dice_score}')
    DSC.append(dice_score)
    TP, TN, FP, FN, recall, precision, FPR, FNR, PWC, Jaccard, Dice = other_eval_metrics(
        gt[:, :, start_slice:end_slice], automated[:, :, start_slice:end_slice])
    stats = {'bone_mask_image': os.path.split(eval_gt_path[i])[-1].split('_dicom')[0],
             'TP': TP,
             'TN': TN,
             'FP': FP,
             'FN': FN,
             'recall': recall,
             'precision': precision,
             'FPR': FPR,
             'FNR': FNR,
             'PWC': PWC,
             'Jaccard': Jaccard,
             'Dice': Dice
             }
    statistics.append(stats)

csv_columns = ['input_image', 'bone_mask_image', 'TP', 'TN', 'FP', 'FN', 'recall', 'precision', 'FPR', 'FNR', 'PWC',
               'Jaccard', 'Dice']
csv_file = os.path.join(data_root, which_dataset, network, test_folder, 'bone_segmentation_stats.csv')
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns, lineterminator='\n')
        writer.writeheader()
        for data in statistics:
            writer.writerow(data)
    print(f"CSV file '{csv_file}' generated successfully.")
except IOError:
    print("I/O error")

csv_file_path = os.path.join(data_root, which_dataset, network, test_folder, 'dsc_eval.csv')
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Image Name', 'DSC'])
    for eval_path, dsc_value in zip(eval_gt_path, DSC):
        image_name = os.path.split(eval_path)[-1].split('_dicom')[0]
        csv_writer.writerow([image_name, dsc_value])

print(f"CSV file '{csv_file_path}' generated successfully.")

print(np.mean(DSC[0:3]))
print(np.mean(DSC[3::]))