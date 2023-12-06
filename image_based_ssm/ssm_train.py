# ssm_train.py

from ssm_model import RecSSM
from ssm_utils import *
from ssm_config import *
import ants
import os
import pickle
import time



def train_ssm(bone, leg_list, fixed_anatomy, model, warped_train_dir, results_dir):

    print('warping training images...')
    if 'femur' in bone:
        label = 1
    else:
        label = 2
    
    if not os.path.exists(os.path.join(results_dir, 'registration_time.csv')):
        with open(os.path.join(results_dir, 'registration_time.csv'), 'w') as f:
            f.write('FILENAME,MINUTES,SECONDS\n')
        
    for filename_tr in leg_list:
        moving_img = ants.image_read(filename_tr)
        moving_mask_img = ants.utils.get_mask(moving_img, label, label+1)
        moving_left_femur = ants.utils.crop_image(moving_mask_img, moving_mask_img)
        
        start_time = time.time() 
        outs = model.reg(fixed_anatomy, moving_left_femur)
        end_time = time.time() 
        minutes, seconds = compute_efficiency(start_time, end_time)
        # print(f"Training Elapsed time: {minutes} minutes {seconds} seconds")
        
        with open(os.path.join(results_dir, 'registration_time.csv'), 'a') as f:
            f.write(f'{os.path.basename(filename_tr)},{minutes},{seconds}\n')
            
        warped_img = outs['warpedmovout']
        ants.image_write(warped_img, warped_train_dir + os.path.basename(filename_tr))
        # print(os.path.basename(filename_tr) + ' is done')
    print('registration of training images is done')

    print('building SSM...')
    
    train_start = time.time()
    model.ssm_train(warped_train_dir)
    train_finish = time.time()
    minutes, seconds = compute_efficiency(train_start, train_finish)
    # print(f"SSM Training Elapsed time: {minutes} minutes {seconds} seconds")  
    with open(os.path.join(results_dir, 'registration_time.csv'), 'a') as f:
        f.write(f'SSM,{minutes},{seconds}\n')
        
    with open(os.path.join(results_dir, 'ssm_model.pkl'), 'wb') as f:
        pickle.dump(model, f)

