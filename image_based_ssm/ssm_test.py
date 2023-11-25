# ssm_test.py 
import os
import pickle
import numpy as np
import ants
import nibabel as nib
from ssm_model import RecSSM
from ssm_utils import *
from ssm_config import *


def test_ssm(model, fixed_anatomy, testing_dir, warped_test_dir, results_dir):

  print('loading SSM...')
  with open(os.path.join(results_dir, 'ssm_model.pkl'), 'rb') as f:
      model = pickle.load(f)


  print('warping test images...')
  for filename_ts in os.listdir(testing_dir):
      moving_img = ants.image_read(os.path.join(testing_dir, filename_ts))
      outs = model.reg(fixed_anatomy, moving_img)
      warped_img = outs['warpedmovout']
      ants.image_write(warped_img, warped_test_dir + filename_ts)
      print(filename_ts + ' is done')


  print('fitting...')
  for filename_wrap in os.listdir(warped_test_dir):
      test = nib.load(os.path.join(warped_test_dir, filename_wrap))
      test_data = test.get_fdata().astype(np.int16)
      rec = model.ssm_test(test_data, useOnlyMeanShape=False)
      implant = rec-test_data

      rec_img = nib.Nifti1Image(rec, test.affine, test.header)
      nib.save(rec_img, os.path.join(results_dir, 'rec_'+filename_wrap))
      implant_img = nib.Nifti1Image(implant, test.affine, test.header)
      nib.save(implant_img, os.path.join(results_dir, 'implant_'+filename_wrap))


      print('converting the results back to original image space...')
      for filename_wrap in os.listdir(testing_dir):
          moving = ants.image_read(os.path.join(testing_dir, filename_wrap))
          moving_rec = ants.image_read(os.path.join(results_dir, 'rec_'+filename_wrap))
          converted_img = model.inverse_reg(fixed_anatomy, moving, moving_rec)
          moving_nib = nib.load(os.path.join(testing_dir, filename_wrap))
          restore_img = nib.Nifti1Image(converted_img.astype(np.int16), moving_nib.affine, moving_nib.header)
          nib.save(restore_img, os.path.join(results_dir, filename_wrap))
