# calculating the DSC and RMSE for the test data

from weakref import ref
import numpy as np
from ssm_utils import *
from ssm_config import *
import nibabel as nib
import nibabel.processing as nibproc
import SimpleITK as sitk
from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import mean_squared_error


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.00001
    return (2. * intersection ) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)






for bone in BONE_STRUCTURE:
    print(f'Processing {bone}...')
    if not os.path.exists(os.path.join(OUT_DIR, bone, 'eval_metric_resampled.csv')):
        with open(os.path.join(OUT_DIR, bone, 'eval_metric_resampled.csv'), 'w') as f:
            f.write('BONE,FILENAME,INPUT TYPE,DSC,HDmax,HD95,HDrmse,RMSE\n')

    for cond in CONDITION:
        print(f'The input data uses {cond}...')

        test_dir = TEST_DIRS[bone]
        results_dir = RESULTS_DIRS_TESTING[bone][cond]
        surgical_plan_dir = test_dir + '_intermediate'

        for filename in os.listdir(test_dir):
            print(f'Processing {filename}')
            ref_nib = nib.load(os.path.join(surgical_plan_dir, filename))
            resampled_ref_nib = nibproc.resample_to_output(ref_nib, (1, 1, 1))
            
            # find the first slice number with non-zero pixel for distal femur computation
            # slice_num = 0
            # while np.sum(resampled_ref_nib.get_fdata()[:,:,slice_num]) == 0:
            #     slice_num += 1
            
            ref_img = resampled_ref_nib.get_fdata()#[:,:,slice_num:(slice_num+150)]
            ref_img[ref_img < 0.5] = 0
            ref_img[ref_img > 0.5] = 1

            rec_nib = nib.load(os.path.join(results_dir, filename))
            resampled_rec_nib = nibproc.resample_to_output(rec_nib, (1, 1, 1))
            rec_img = resampled_rec_nib.get_fdata()#[:,:,slice_num:(slice_num+150)]
            rec_img[rec_img < 0.5] = 0
            rec_img[rec_img > 0.5] = 1

            # computing DSC
            dsc = dice_coef(ref_img, rec_img)
            # print(f'the DSC is {dsc}')

            # computing max Hausdorff distance between the reference and reconstructed images
            ref_image = sitk.GetImageFromArray(ref_img)
            ref_image_int = sitk.Cast(ref_image, sitk.sitkUInt8)
            ref_surface = sitk.LabelContour(ref_image_int)
            ref_surface_array = sitk.GetArrayFromImage(ref_surface)
            ref_coords = np.array(np.where(ref_surface_array == 1)).T

            rec_image = sitk.GetImageFromArray(rec_img)
            rec_image_int = sitk.Cast(rec_image, sitk.sitkUInt8)
            rec_surface = sitk.LabelContour(rec_image_int)
            rec_surface_array = sitk.GetArrayFromImage(rec_surface)
            rec_coords = np.array(np.where(rec_surface_array == 1)).T

            hausdorff_dist = (directed_hausdorff(ref_coords, rec_coords)[0] + directed_hausdorff(rec_coords, ref_coords)[0])/2
            # print(f'the max Hausdorff distance is {hausdorff_dist}')

            # computing the 95% Hausdorff distance
            # SignedDanielssonDistanceMap/  SignedMaurerDistanceMap selected
            seg_distance_map = sitk.Abs(
                sitk.SignedMaurerDistanceMap(rec_surface, squaredDistance=False, useImageSpacing=True))
            reference_segmentation_distance_map = sitk.Abs(
                sitk.SignedMaurerDistanceMap(ref_surface, squaredDistance=False, useImageSpacing=True))

            dist_seg = sitk.GetArrayViewFromImage(seg_distance_map)[
                sitk.GetArrayViewFromImage(ref_surface) == 1]
            dist_ref = sitk.GetArrayViewFromImage(reference_segmentation_distance_map)[
                sitk.GetArrayViewFromImage(rec_surface) == 1]
            hd_95 = (np.percentile(dist_ref, 95) + np.percentile(dist_seg, 95)) / 2.0
            # print(f'the 95% Hausdorff distance is {hd_95}')
            HD_rmse = (np.sqrt(np.mean(dist_ref ** 2)) + np.sqrt(np.mean(dist_seg ** 2))) / 2.0
            # print(f'the Hausdorff RMSE is {HD_rmse}')

            # compute RMSE
            rmse = np.sqrt(mean_squared_error(ref_img, rec_img))
            # print(f'the RMSE is {rmse}')

            # saving difference image
            # diff = ref_img -rec_img
            # diff_img = nib.Nifti1Image(diff, ref_nib.affine, ref_nib.header)
            # nib.save(diff_img, os.path.join(results_dir, 'diff_'+filename))

            # save RMSE and DSC to csv file
            with open(os.path.join(OUT_DIR, bone, 'eval_metric_resampled.csv'), 'a') as f:
                f.write(f'{bone},{filename},{cond},{dsc},{hausdorff_dist},{hd_95},{HD_rmse},{rmse}\n')
