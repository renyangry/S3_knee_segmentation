# ssm_test_suegical_plan.py 

import os
import pickle
import numpy as np
import ants
import nibabel as nib
import cv2
from ssm_model import RecSSM
from ssm_utils import *
from ssm_config import *


def test_ssm_surgical_plan(model_dir, fixed_anatomy, testing_dir, warped_test_dir, results_dir, bone, condition):

    print('loading SSM...')
    with open(os.path.join(model_dir, 'ssm_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    intermediate_dir = testing_dir + '_intermediate'
    os.makedirs(intermediate_dir, exist_ok=True)

    print('warping test images...')
    if 'femur' in bone:
        label = 1
    else:
        label = 2

    for filename_ts in os.listdir(testing_dir):
        moving_img = ants.image_read(os.path.join(testing_dir, filename_ts))
        moving_img_spacing = moving_img.spacing
        volume_slice = int(150/moving_img_spacing[2])
        moving_mask_img = ants.utils.get_mask(moving_img, label, label + 1)
        moving_left_femur = ants.utils.crop_image(moving_mask_img, moving_mask_img)
        ## for surgical plan and mesh SSM testing
        ants.image_write(moving_left_femur, os.path.join(intermediate_dir,
                                                         filename_ts))
        ##
        lf_shape = moving_left_femur.shape

        if condition == 'proximal_only':
            print('extract 150mm femur head volume from test images...')
            moving_femur_head = ants.utils.crop_indices(moving_left_femur, (0, 0, lf_shape[2]-volume_slice), lf_shape)
            ## for surgical plan and mesh SSM testing
            ants.image_write(moving_femur_head, os.path.join(intermediate_dir,
                                                             'po_'+filename_ts))
            ##
            outs = model.reg(fixed_anatomy, moving_femur_head)

        # need to consider how much volume is needed from the distal femur - add code here
        # activate the condition below if you want to preserve the distal femur volume
        if condition == 'add_distal':
            print('preserve the 150mm proximal + distal femur volume...')
            moving_femur = moving_left_femur.numpy()
            moving_femur[:, :, volume_slice:-volume_slice] = 0
            moving_femur_volume = ants.from_numpy(moving_femur)
            ## for surgical plan and mesh SSM testing
            ants.image_write(moving_femur_volume, os.path.join(intermediate_dir,
                                                             'ad_' + filename_ts))
            ##
            outs = model.reg(fixed_anatomy, moving_femur_volume)

        if condition == 'reduced_distal':
            print('preserve the 150mm proximal + reduced distal femur volume...')
            moving_femur = moving_left_femur.numpy()
            moving_femur[:, :, volume_slice:-volume_slice] = 0

            distal_femur = moving_femur[:, :, 0:volume_slice]
            distal_offset = int(5 / moving_img_spacing[0])
            print('the total pixel to be pushed in is: ' + str(distal_offset))
            # erode the distal_femur in with xy_offset pixels
            kernel = np.ones((distal_offset, distal_offset), np.uint8)
            new_distal_femur = cv2.erode(distal_femur, kernel)
            moving_femur[:, :, 0:volume_slice] = new_distal_femur
            moving_femur_volume = ants.from_numpy(moving_femur)
            ## for surgical plan and mesh SSM testing
            ants.image_write(moving_femur_volume, os.path.join(intermediate_dir,
                                                               'rd_' + filename_ts))
            ##
            outs = model.reg(fixed_anatomy, moving_femur_volume)


        warped_img = outs['warpedmovout']
        ants.image_write(warped_img, warped_test_dir + filename_ts)
        print(filename_ts + ' is done')


    print('fitting...')
    for filename_wrap in os.listdir(warped_test_dir):
        test = nib.load(os.path.join(warped_test_dir, filename_wrap))
        test_data = test.get_fdata().astype(np.int16)
        rec = model.ssm_test(test_data, useOnlyMeanShape=False)
        rec[rec < 0.5] = 0
        rec[rec >= 0.5] = 1
        rec_img = nib.Nifti1Image(rec, test.affine, test.header)
        nib.save(rec_img, os.path.join(results_dir, 'rec_'+filename_wrap))

        ## implant = rec-test_data
        ## implant_img = nib.Nifti1Image(implant, test.affine, test.header)
        ## nib.save(implant_img, os.path.join(results_dir, 'implant_'+filename_wrap))


    print('converting the results back to original image space...')
    for filename_wrap in os.listdir(testing_dir):
        moving = ants.image_read(os.path.join(intermediate_dir, filename_wrap))
        moving_rec = ants.image_read(os.path.join(results_dir, 'rec_'+filename_wrap))
        converted_img = model.inverse_reg(fixed_anatomy, moving, moving_rec)
        moving_nib = nib.load(os.path.join(intermediate_dir, filename_wrap))
        restore_img = nib.Nifti1Image(converted_img.astype(np.int16), moving_nib.affine, moving_nib.header)
        nib.save(restore_img, os.path.join(results_dir, filename_wrap))

    del model
    gc.collect()