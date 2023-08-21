import os
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi


def global_threshold_mask(raw_img, lower_threshold, upper_threshold):
    mask = np.zeros_like(raw_img)
    mask[(raw_img > lower_threshold) & (raw_img < upper_threshold)] = 1
    return mask


def generate_body_mask(threshold_mask):
    body_mask = np.zeros_like(threshold_mask)
    for i in range(threshold_mask.shape[2]):
        label_objects, nb_labels = ndi.label(threshold_mask[:, :, i])
        sizes = np.bincount(label_objects.ravel())
        sizes[0] = 0
        mask_sizes = (sizes > np.percentile(sizes, 99))
        body_mask[:, :, i] = mask_sizes[label_objects]
    return body_mask


def restore_air_signal(body_mask):
    body_mask = body_mask.astype(bool)
    bkg_mask = np.multiply(~body_mask, -1024)
    return bkg_mask


def remove_ct_couch(raw_img, body_mask):
    intermediate = np.multiply(raw_img, body_mask)
    bkg_mask = restore_air_signal(body_mask)
    pt_ct_img = np.add(intermediate, bkg_mask)
    return pt_ct_img


def nii_file_writer(raw_img_path, img2save, prefix, path4output):
    'prefix need to be in string format'
    affine_im = nib.load(raw_img_path).affine
    header_im = nib.load(raw_img_path).header
    metadata = nib.Nifti1Image(img2save, affine=affine_im, header=header_im)
    file_name = '_'.join([prefix, os.path.split(raw_img_path)[-1]])
    nib.save(metadata, os.path.join(path4output, file_name))


def split_img(img_mtx):
    right_img = img_mtx[0:(img_mtx.shape[0] // 2), :, :]
    left_img = img_mtx[(img_mtx.shape[0] // 2)::, :, :]
    return right_img, left_img


def reconstruct_img(right_img, left_img):
    return np.vstack((right_img, left_img))


def check_path_exist(path_):
    if not os.path.exists(path_):
        os.makedirs(path_)
        print(f"Path '{path_}' created.")
    else:
        print(f"Path '{path_}' already exists.")


def check_num_of_pt_4eval(manual_path, automated_path):
    try:
        assert len(manual_path) == len(automated_path)
        print("Pass: Detected same number of segmentations in two folders.")
    except AssertionError:
        print("Error: The number of segmentations doesn't match in two folders.")
