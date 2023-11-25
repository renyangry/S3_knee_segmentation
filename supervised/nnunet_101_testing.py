from collections import OrderedDict
import glob
import shutil
import sys
import os
import json
sys.path.append('/home/rgu/Documents/GitHub/S3_knee_segmentation/ct_all_preprocess')
import numpy as np


from pathlib import Path
from typing import List
import json
import glob
import re
from utils import check_path_exist



data_root = '/home/rgu/Documents/'
which_dataset = 'UK_dataset'


leg_ct_dir = sorted(glob.glob(os.path.join(data_root, which_dataset, 'seperated_leg_img', "*.nii.gz")))
testing_leg_ct_dir = np.random.choice(leg_ct_dir, 100, replace=False)
testing_leg_ct_list = testing_leg_ct_dir.tolist()
testing_dataset = {'test': testing_leg_ct_list}

remaining_leg_ct_dir = [item for item in leg_ct_dir if item not in testing_leg_ct_dir]
redundant_dataset = {'others': remaining_leg_ct_dir}
with open(os.path.join(data_root, which_dataset, 'nnUNet_testing', 'testing_dataset.json'), 'w') as f:
    json.dump(testing_dataset, f)
    f.write('\n')  
    json.dump(redundant_dataset, f)


test_image_dir = os.path.join(data_root, which_dataset, 'nnUNet_testing')

for file in testing_leg_ct_dir:
    filename = file.split('/')[-1]
    if filename.startswith('left_'):
        case_num = filename.split('/')[-1].split('BR-')[1]
        new_name = 'halvleg_1' + case_num + '_0000.nii.gz'
        print(f'Copying {filename} to {new_name}')
        shutil.copy(file, os.path.join(test_image_dir, new_name)) # type: ignore
    elif filename.startswith('right_'):
        case_num = filename.split('/')[-1].split('BR-')[1]
        new_name = 'halvleg_' + case_num + '_0000.nii.gz'
        print(f'Copying {filename} to {new_name}')
        shutil.copy(file, os.path.join(test_image_dir, new_name)) # type: ignore
    else:
        print(f'Error: {filename} is not start with left_ or right_')
        sys.exit()


