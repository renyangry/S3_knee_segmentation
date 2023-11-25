import ants
import os
from matplotlib.pylab import f
import matplotlib.pyplot as plt
import numpy as np
import cv2
from ssm_config import *

bone = 'left_femur'
testing_dir = TEST_DIRS[bone]
nii_files = [f.name for f in os.scandir(testing_dir) if f.name.endswith('.nii.gz')]

# for condition in CONDITION:
#     print('processing ' + condition + ' images...')
#     for filename_ts in nii_files:
#             moving_img = ants.image_read(os.path.join(testing_dir, filename_ts))
#             moving_img_spacing = moving_img.spacing
#             volume_slice = int(150/moving_img_spacing[2])
#             moving_left_femur = ants.utils.crop_image(moving_img, label=1)
#             ants.image_write(moving_left_femur, os.path.join(testing_dir, 'intermediate', str('cropped_')+filename_ts))
#             lf_shape = moving_left_femur.shape
            
#             if condition == 'proximal_only':
#                 moving_femur = moving_left_femur.numpy()
#                 moving_femur[:, :, 0:-volume_slice] = 0
#                 moving_femur_volume = ants.from_numpy(moving_femur)
#                 ants.image_write(moving_femur_volume, os.path.join(testing_dir, 'intermediate', str('proximal_only_')+filename_ts))
                
#             if condition == 'add_distal':
#                 moving_femur = moving_left_femur.numpy()
#                 moving_femur[:, :, volume_slice:-volume_slice] = 0
#                 moving_femur_volume = ants.from_numpy(moving_femur)
#                 ants.image_write(moving_femur_volume, os.path.join(testing_dir, 'intermediate', str('add_distal_')+filename_ts))
                
#             if condition == 'reduced_distal':
#                 moving_femur = moving_left_femur.numpy()
#                 moving_femur[:, :, volume_slice:-volume_slice] = 0
                
#                 distal_femur = moving_femur[:, :, 0:volume_slice]
#                 distal_offset = int(10 / moving_img_spacing[0])
#                 print('the total pixel to be pushed in is: ' + str(distal_offset))
#                 # erode the distal_femur in with xy_offset pixels 
#                 kernel = np.ones((distal_offset, distal_offset), np.uint8)
#                 new_distal_femur = cv2.erode(distal_femur, kernel)
#                 moving_femur[:, :, 0:volume_slice] = new_distal_femur
#                 moving_femur_volume = ants.from_numpy(moving_femur)
#                 ants.image_write(moving_femur_volume, os.path.join(testing_dir, 'intermediate', str('reduced_distal_')+filename_ts))


warped_dir = WARPED_DIRS[bone]['train']
fixed_anatomy = ants.image_read(ROOT_DIR + bone + '.nii.gz')
midslice_idx = int(fixed_anatomy.shape[1] / 2)

f, ax = plt.subplots(23, 4, figsize=(55, 55))
idx = 0
for i in range(23):
    for j in range(4):
        filename_wp = os.listdir(warped_dir)[idx]
        wrapped_anatomy = ants.image_read(os.path.join(warped_dir, filename_wp))

        # image = fixed_anatomy.numpy()[:, midslice_idx, :]
        # overlay = wrapped_anatomy.numpy()[:, midslice_idx, :]

        image = fixed_anatomy.numpy()[:, 85, :]
        overlay = wrapped_anatomy.numpy()[:, 85, :]
        
        overlay[overlay < 0.3] =np.nan

        ax[i, j].imshow(image, cmap='gray', aspect='equal')
        ax[i, j].imshow(overlay, alpha=0.5, cmap='OrRd', aspect='equal')
        ax[i, j].axis('off')
        idx += 1

plt.tight_layout()
plt.savefig(os.path.join(ROOT_DIR, 'plot.png'))
plt.show()


