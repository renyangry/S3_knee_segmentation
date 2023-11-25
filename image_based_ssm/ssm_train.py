# ssm_train.py

from ssm_model import RecSSM
from ssm_utils import *
from ssm_config import *
import ants
import os
import pickle


def train_ssm(bone, leg_list, fixed_anatomy, model, warped_train_dir, results_dir):

    print('warping training images...')
    if 'femur' in bone:
        label = 1
    else:
        label = 2
        
    for filename_tr in leg_list:
        moving_img = ants.image_read(filename_tr)
        moving_mask_img = ants.utils.get_mask(moving_img, label, label+1)
        moving_left_femur = ants.utils.crop_image(moving_mask_img, moving_mask_img)
        outs = model.reg(fixed_anatomy, moving_left_femur)
        warped_img = outs['warpedmovout']
        ants.image_write(warped_img, warped_train_dir + os.path.basename(filename_tr))
        print(os.path.basename(filename_tr) + ' is done')


    print('building SSM...')
    model.ssm_train(warped_train_dir)
    with open(os.path.join(results_dir, 'ssm_model.pkl'), 'wb') as f:
        pickle.dump(model, f)

