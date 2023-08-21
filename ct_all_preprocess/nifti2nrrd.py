import os
import nrrd
import nibabel as nib
import glob
import logger
from collections import OrderedDict
import numpy as np
from utils import check_path_exist


def nii2nrrd(nifti_img_path,output_path,out_fn=None):
    if not nifti_img_path.endswith('.nii.gz'):
        logger.info('skipped')
    else:
        image = nib.load(nifti_img_path)
        # nib.aff2axcodes(image.affine)
        img_spc = image.affine[0:-1,-1]
        img_spc[0:2] *= -1
        nrrd_header_dict = OrderedDict([
            ('type', 'short'),
            ('dimension', len(image.shape)),
            ('space', 'left-posterior-superior'),
            ('sizes', np.array(image.shape)),
            ('space directions', np.array(image.affine[0:3,0:3])),
            ('kinds', ['domain', 'domain', 'domain']),
            ('endian', 'little'),
            ('encoding', 'raw'),
            ('space origin', img_spc)
        ])
        if out_fn is None:
            out_fn = (os.path.split(nifti_img_path)[1]).replace('nii.gz','nrrd')
        output = np.flip(image.get_fdata(),axis=(0,1))
        nrrd.write(os.path.join(output_path,out_fn),output,nrrd_header_dict)


data_root = '/home/rgu/Documents/'
which_dataset = 'UK dataset'
image_path = sorted(glob.glob(os.path.join(data_root,which_dataset,'seperated_leg_img','*.nii.gz')))
nrrd_output_path = os.path.join(data_root,which_dataset,'seperated_leg_nrrd')
check_path_exist(nrrd_output_path)
for pt in range(len(image_path)):
    nii2nrrd(image_path[pt],nrrd_output_path,None)
    # print(' '.join([os.path.split(image_path[pt])[1],'done']))

print('Done')


