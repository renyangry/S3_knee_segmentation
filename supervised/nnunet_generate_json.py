from collections import OrderedDict
import glob
import shutil
import sys
import os
import json
sys.path.append('/home/rgu/Documents/GitHub/S3_knee_segmentation/ct_all_preprocess')


from pathlib import Path
from typing import List
import json
import glob
import re
from utils import check_path_exist



def rename_img_file_nnunet(image_dir):
    for filename in os.listdir(image_dir):
        if filename.startswith('left_'):
            case_num = filename.split('/')[-1].split('BR-')[1]
            new_name = 'halvleg_1' + case_num + '_0000.nii.gz'
            print(f'Renaming {filename} to {new_name}')
            os.rename(os.path.join(image_dir, filename),
                  os.path.join(image_dir, new_name))
        elif filename.startswith('right_'):
            case_num = filename.split('/')[-1].split('BR-')[1]
            new_name = 'halvleg_' + case_num + '_0000.nii.gz'
            print(f'Renaming {filename} to {new_name}')
            os.rename(os.path.join(image_dir, filename),
                  os.path.join(image_dir, new_name))
        else:
            print(f'Error: {filename} is not start with left_ or right_')
            sys.exit()

def rename_lab_file_nnunet(image_dir):
    for filename in os.listdir(image_dir):
        if 'left_' in filename:
            case_num = filename.split('/')[-1].split('BR-')[1]
            new_name = 'halvleg_1' + case_num + '.nii.gz'
            print(f'Renaming {filename} to {new_name}')
            os.rename(os.path.join(image_dir, filename),
                  os.path.join(image_dir, new_name))
        elif 'right_' in filename:
            case_num = filename.split('/')[-1].split('BR-')[1]
            new_name = 'halvleg_' + case_num + '.nii.gz'
            print(f'Renaming {filename} to {new_name}')
            os.rename(os.path.join(image_dir, filename),
                  os.path.join(image_dir, new_name))
        else:
            print(f'Error: {filename} is not start with left_ or right_')
            sys.exit()

data_root = '/home/rgu/Documents/'
which_dataset = 'UK dataset'
training_dataset = 'Dataset104_ps'

leg_ct_dir = os.path.join(data_root, which_dataset, 'seperated_leg_img')
train_label_dir = os.path.join(data_root, which_dataset, 'nnUNet_raw', training_dataset, 'labelsTr')
train_image_dir = os.path.join(data_root, which_dataset, 'nnUNet_raw', training_dataset, 'imagesTr')
# rename_lab_file_nnunet(train_label_dir)
# rename_img_file_nnunet(train_image_dir)

train_image = sorted(glob.glob(os.path.join(train_image_dir, "*.nii.gz")))
train_label = sorted(glob.glob(os.path.join(train_label_dir, "*.nii.gz")))

train_image = ["{}".format(item.split('/')[-1]) for item in train_image]
train_label = ["{}".format(item.split('/')[-1]) for item in train_label]


# creating testing dataset 
# test_left_image_dir = os.path.join(data_root, which_dataset, 'nnUNet_testing')
# rename_img_file_nnunet(test_left_image_dir)

# for filename in os.listdir(leg_ct_dir):
#     if filename.startswith('left_'):
#         case_num = filename.split('/')[-1].split('BR-')[1]
#         new_name = 'halvleg_' + case_num + '_0000.nii.gz'
#         print(f'Copying {filename} to {new_name}')
#         shutil.copy(os.path.join(leg_ct_dir, filename), os.path.join(test_left_image_dir, new_name))
#     elif filename.startswith('right_'):
#         case_num = filename.split('/')[-1].split('BR-')[1]
#         new_name = 'halvleg_' + case_num + '_0000.nii.gz'
#         print(f'Copying {filename} to {new_name}')
#         shutil.copy(os.path.join(leg_ct_dir, filename), os.path.join(test_right_image_dir, new_name))
#     else:
#         print(f'Error: {filename} is not start with left_ or right_')
#         sys.exit()

# test_image_dir = [test_left_image_dir, test_right_image_dir]
# test_image = []
# for dir in test_image_dir:
    # test_image += sorted(glob.glob(os.path.join(dir, "*.nii.gz")))
# test_image = ["{}".format(item.split('/')[-1]) for item in test_image]



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
    "partial_label": "3",
    }

json_dict['numTraining'] = len(train_image)
# json_dict['numTest'] = len(test_image)
json_dict['training'] = [{'image': "./imagesTr/%s" % i, "label": "./labelsTr/%s" % i} for i in train_label]
# json_dict['test'] = ["./imagesTs/%s" % i for i in test_image]
with open(os.path.join(data_root, which_dataset, 'nnUNet_raw', training_dataset, "dataset.json"), 'w') as f:
    json.dump(json_dict, f)
