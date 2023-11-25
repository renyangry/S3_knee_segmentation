import os
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from scipy.ndimage import label


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


def bone_seg_separation(img, seg_value):
    mask_img = np.zeros_like(img)
    if seg_value is None:
        return mask_img
    else:
        mask_img[img == seg_value] = 1
        return mask_img

def clean_up_seg(seg_img,size_threshold):
    labelled_volume, num_features = label(seg_img)  
    if num_features > 1:    
        sizes = np.bincount(labelled_volume.ravel())
        bone_sizes: Optional[bool] = (sizes > size_threshold)
        bone_sizes[0] = 0
        seg_img = bone_sizes[labelled_volume]
    labelled_volume1, num_features1 = label(seg_img)
    print(f"Number of connected components of cleaned segmentation: {num_features1}")
    return seg_img

def single_bone_seg_process(seg_img, threshold_list, idx_list, output_path, prefix):
    for idx in idx_list:
        seg = nib.load(seg_img[idx]).get_fdata()
        right_seg, left_seg = split_img(seg)

        if 'left' in prefix: 
            single_bone = bone_seg_separation(left_seg, threshold_list[idx])
        elif 'right' in prefix:
            single_bone = bone_seg_separation(right_seg, threshold_list[idx])
        if 'tibia' in prefix: 
            cleaned_bone = clean_up_seg(single_bone, 1000) * 2
        else: 
            cleaned_bone = clean_up_seg(single_bone, 1000)
        nii_file_writer(seg_img[idx], cleaned_bone, prefix, output_path)