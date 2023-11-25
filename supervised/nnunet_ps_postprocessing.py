from glob import glob
import os 
import nibabel as nib
import numpy as np



testing_ps_path = '/home/rgu/Documents/UK dataset/nnUNet_raw/testing_ps'
testing_label = sorted(glob(os.path.join(testing_ps_path,'*.nii.gz')))
# testing_dual_label_path = '/home/rgu/Documents/UK dataset/nnUNet_raw/testing_dual_label'
# testing_dual_label = os.path.join(testing_dual_label_path,'*.nii.gz')


for i in range(len(testing_label)):
    label = nib.load(testing_label[i]).get_fdata() # type: ignore
    label = np.rint(label).astype(np.int8)
    label[label == 3] = np.int8(0)
    metadata = nib.Nifti1Image(label, affine=nib.load(testing_label[i]).affine, header=nib.load(testing_label[i]).header)
    nib.save(metadata,testing_label[i])
    print('Processing: ', testing_label[i])

print('Done!')



    

    

