import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting
from numpy import disp


file_path = '/home/rgu/Documents/test/left_femur/femur_left_BR-119BR-119_dicom.nii.gz' 
overlay_path = '/home/rgu/Documents/ssm_results/left_femur/proximal_only/femur_left_BR-119BR-119_dicom.nii.gz'
vol = nib.load(file_path)#.get_fdata()
overlay = nib.load(overlay_path)#.get_fdata()

fig, ax = plt.subplots(figsize=[10, 5])
# display = plotting.plot_img(vol, cmap='gray', axes=ax)
# display.add_overlay(overlay_path, cmap='Reds_r')
# plt.show()

display = plotting.plot_img(vol, display_mode='mosaic', cmap='gray')
display.add_contours(overlay, colors="r", linewidths=0.5)
plt.show()
