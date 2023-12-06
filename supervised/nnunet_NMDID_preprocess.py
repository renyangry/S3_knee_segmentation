from collections import OrderedDict
import glob
import sys
import os
import json
import shutil
sys.path.append('/home/rgu/Documents/GitHub/S3_knee_segmentation/ct_all_preprocess')
from utils import *


def single_leg_seg(side_seg, femur_threshold, tibia_threshold): 
    femur = bone_seg_separation(side_seg, femur_threshold)
    tibia = bone_seg_separation(side_seg, tibia_threshold)
    cleaned_femur = clean_up_seg(femur, 1000)
    cleaned_tibia = clean_up_seg(tibia, 1000) * 2
    left_leg = cleaned_femur + cleaned_tibia
    return left_leg

def rename_img_file_nnunet(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.startswith('left_'):
            case_num = filename.split('_')[1]
            new_name = 'halvleg_2' + case_num[1::] + '_0000.nii.gz'
            print(f'Renaming {filename} to {new_name}')
            os.rename(os.path.join(input_dir, filename),
                  os.path.join(output_dir, new_name))
        elif filename.startswith('right_'):
            case_num = filename.split('_')[1]
            new_name = 'halvleg_' + case_num + '_0000.nii.gz'
            print(f'Renaming {filename} to {new_name}')
            os.rename(os.path.join(input_dir, filename),
                  os.path.join(output_dir, new_name))
        else:
            print(f'Error: {filename} is not start with left_ or right_')
            sys.exit()

def rename_lab_file_nnunet(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if 'left_' in filename:
            case_num = filename.split('_')[1]
            new_name = 'halvleg_2' + case_num[1::] + '.nii.gz'
            print(f'Renaming {filename} to {new_name}')
            os.rename(os.path.join(input_dir, filename),
                  os.path.join(output_dir, new_name))
        elif 'right_' in filename:
            case_num = filename.split('_')[1]
            new_name = 'halvleg_' + case_num + '.nii.gz'
            print(f'Renaming {filename} to {new_name}')
            os.rename(os.path.join(input_dir, filename),
                  os.path.join(output_dir, new_name))
        else:
            print(f'Error: {filename} is not start with left_ or right_')
            sys.exit()

data_root = '/home/rgu/Documents/'
which_dataset = 'new_mexico'
training_dataset = 'Dataset1000_NMDID'
print(f'Processing {which_dataset} dataset')
leg_ct_dir = sorted(glob.glob(os.path.join(data_root, which_dataset, '*_ct.nii.gz')))
train_label_dir = sorted(glob.glob(os.path.join(data_root, which_dataset, '*_label.nii.gz')))
output_path_side = os.path.join(data_root,which_dataset,'seperated_leg_img')
check_path_exist(output_path_side)
output_path_seg = os.path.join(data_root,which_dataset,'seperated_leg_seg')
check_path_exist(output_path_seg)
print(f'Output path: {output_path_side}')
print(f'Output path: {output_path_seg}')

# for idx in range(len(leg_ct_dir)):
#     img = nib.load(leg_ct_dir[idx]).get_fdata()    
#     right_img, left_img = split_img(img)
#     nii_file_writer(leg_ct_dir[idx],right_img,'right',output_path_side)
#     nii_file_writer(leg_ct_dir[idx],left_img,'left',output_path_side)
# print('Image seperation done')

# right_femur_threshold = [2088, 3309, 2088, 2088, 2088, 2088, 2088, 2088, 2088, 2088, 2088, 2088, 3309, 3696, 2088, 2088, 2088, 2088, 2088, 2088, 2088, 2088, 2088, 1637, 2088, 2435, 2088, 2088, 2088, 2088, 1637, 2088, 2088, 2088, 3309, 2238, 2008, 2088, 2088, 2088, 2088, 2088, 2088, 2088, 2088, 2088, 2088, 2088, 2088, 2088, 3009, 2008, 2008, 2008, 2008]
# left_femur_threshold = [3309, 2131, 2603, 2603, 2603, 2603, 2603, 2603, 3009, 2603, 3696, 3696, 2131, 2283, 2603, 2603, 2603, 3696, 2603, 3696, 2603, 2603, 2603, 2283, 2603, 3309, 2435, 2603, 3696, 2603, 2131, 2603, 2603, 3309, 2131, 2603, 2603, 2603, 3696, 2603, 2603, 2603, 3696, 2603, 2603, 2603, 2603, 2603, 2603, 2603, 2435, 2603, 2603, 2603, 2969]
# right_tibia_threshold = [2238, 3009, 2435, 3009, 2131, 3309, 3309, 2435, 2131, 2969, 3309, 3309, 3009, 2435, 3009, 3009, 2435, 3309, 3009, 3309, 3009, 3009, 3009, 3009, 2435, 3244, 2238, 2435, 3309, 3009, 2088, 3009, 2435, 2899, 3009, 3009, 3244, 3009, 3309, 3009, 3009, 3009, 3309, 2283, 2435, 3309, 2435, 3009, 2435, 3009, 2088, 3009, 3309, 2283, 2435]
# left_tibia_threshold = [2435, 2435, 3309, 3244, 1637, 2131, 2131, 3009, 3244, 3123, 2969, 2131, 2435, 2899, 2436, 2435, 3309, 2283, 2435, 2283, 2435, 2435, 2435, 2435, 3309, 3009, 3009, 3309, 2969, 2435, 2603, 2435, 3009, 2238, 2435, 3244, 2238, 3244, 2283, 2435, 2435, 2435, 2969, 3309, 3309, 2131, 3309, 2435, 3309, 2435, 2603, 2435, 2131, 3244, 2238]
# assert len(train_label_dir) == len(right_femur_threshold) == len(left_femur_threshold) == len(right_tibia_threshold) == len(left_tibia_threshold)
# print('Thresholds are ready')

# for i in range(len(train_label_dir)):
#     seg_img = nib.load(train_label_dir[i]).get_fdata()
#     right_seg, left_seg = split_img(seg_img)
#     left_leg = single_leg_seg(left_seg, left_femur_threshold[i], left_tibia_threshold[i])
#     nii_file_writer(train_label_dir[i], left_leg, 'left', output_path_seg)
#     print(f"Case {train_label_dir[i].split('/')[-1]} done")
#     right_leg = single_leg_seg(right_seg, right_femur_threshold[i], right_tibia_threshold[i])
#     nii_file_writer(train_label_dir[i], right_leg, 'right', output_path_seg)
#     print(f"Case {train_label_dir[i].split('/')[-1]} done")


print('preprocessing for nnU-Net training starting here...')
nnunet_dataset_path = os.path.join(data_root, 'UK dataset', 'nnUNet_raw', training_dataset)
# check_path_exist(nnunet_dataset_path)
nnunet_label_dir = os.path.join(data_root, 'UK dataset', 'nnUNet_raw', training_dataset, 'labelsTr')
nnunet_image_dir = os.path.join(data_root, 'UK dataset', 'nnUNet_raw', training_dataset, 'imagesTr')
# check_path_exist(nnunet_label_dir)
# check_path_exist(nnunet_image_dir)
# print(f'Output path: {nnunet_label_dir}')
# print(f'Output path: {nnunet_image_dir}')

# rename_lab_file_nnunet(output_path_seg, nnunet_label_dir)
# rename_img_file_nnunet(output_path_side, nnunet_image_dir)


# 04/12/23: removed because unstable quality of generated segmentation in testing_dual_label
# addtional_label = sorted(glob.glob(os.path.join(data_root, 'UK dataset', 'nnUNet_raw', 'testing_dual_label', '*.nii.gz')))
# addtional_image = sorted(glob.glob(os.path.join(data_root, 'UK dataset', 'nnUNet_testing', '*.nii.gz')))
# for image_file in addtional_image:
#     shutil.copy(image_file, nnunet_image_dir)
# for label_file in addtional_label:
#     shutil.copy(label_file, nnunet_label_dir)
# print('Additional data copied')
##############################################################################################################


train_image = sorted(glob.glob(os.path.join(nnunet_image_dir, "*.nii.gz")))
train_label = sorted(glob.glob(os.path.join(nnunet_label_dir, "*.nii.gz")))
train_image = ["{}".format(item.split('/')[-1]) for item in train_image]
train_label = ["{}".format(item.split('/')[-1]) for item in train_label]

test_image = sorted(glob.glob(os.path.join(data_root, 'UK dataset', 'nnUNet_testing', "*.nii.gz")))
test_image = ["{}".format(item.split('/')[-1]) for item in test_image]

print('Creating json file...')
json_dict = OrderedDict()
json_dict['name'] = "FullLegCT"
json_dict['tensorImageSize'] = "3D"
json_dict['file_ending'] = ".nii.gz"
json_dict['channel_names'] = {
    "0": "CT",
}
json_dict['labels'] = {
    "background": "0",
    "femur": "1",
    "tibia": "2",
    }

json_dict['numTraining'] = len(train_image)
json_dict['numTest'] = len(test_image)
json_dict['training'] = [{'image': "./imagesTr/%s" % i, "label": "./labelsTr/%s" % i} for i in train_label]
json_dict['test'] = ["./imagesTs/%s" % i for i in test_image]
with open(os.path.join(data_root, 'UK dataset', 'nnUNet_raw', training_dataset, "dataset.json"), 'w') as f:
    json.dump(json_dict, f)
print('Json file created')