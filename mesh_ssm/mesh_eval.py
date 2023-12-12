import os 
import glob
from weakref import ref
from pyvista import surface_from_para
import trimesh 
import nibabel as nib
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def mesh2surf(vertices, ref_img, padding=40):
    image_size = padding + np.array(ref_img.shape)
    img = np.zeros(image_size)
    for coordinate in vertices: 
        coordinate = coordinate.astype(int)
        img[tuple(coordinate)] = 1
    image = sitk.GetImageFromArray(img)
    image_int = sitk.Cast(image, sitk.sitkUInt8)
    surface_img = sitk.LabelContour(image_int)
    return surface_img 


bone = 'left_femur'
cond = 'add_distal'
root_dir = "/home/rgu/Documents/new_ssm"
mesh_output_dir = "/home/rgu/Documents/new_ssm/mesh_output"
ref_img_dir = "/home/rgu/Documents/test/left_femur_intermediate"
ref_mesh_dir = "/home/rgu/Documents/new_ssm/simplify_test_stl"

if not os.path.exists(os.path.join(root_dir, 'eval_metric_resampled.csv')):
    with open(os.path.join(root_dir, 'eval_metric_resampled.csv'), 'w') as f:
        f.write('BONE,FILENAME,INPUT TYPE,HDmax1,HDmax2,HD95_1,HD95_2,HDrmse1,HDrmse2\n')
            
for pt in os.listdir(mesh_output_dir):
    mesh_output = trimesh.load(glob.glob(os.path.join(mesh_output_dir, pt, '*.stl'))[0])
    ref_img = nib.load(os.path.join(ref_img_dir, pt + '.nii.gz'))
    ref_mesh = trimesh.load(os.path.join(ref_mesh_dir, pt + '.stl'))
    ref_coords = ref_mesh.vertices
    rec_coords = mesh_output.vertices

    # computing the 95% Hausdorff distance
    ref_surface = mesh2surf(ref_coords, ref_img)
    rec_surface = mesh2surf(rec_coords, ref_img)
    seg_distance_map = sitk.Abs(
        sitk.SignedMaurerDistanceMap(rec_surface, squaredDistance=False, useImageSpacing=True))
    reference_segmentation_distance_map = sitk.Abs(
        sitk.SignedMaurerDistanceMap(ref_surface, squaredDistance=False, useImageSpacing=True))

    dist_seg = sitk.GetArrayViewFromImage(seg_distance_map)[
        sitk.GetArrayViewFromImage(ref_surface) == 1]
    dist_ref = sitk.GetArrayViewFromImage(reference_segmentation_distance_map)[
        sitk.GetArrayViewFromImage(rec_surface) == 1]

    # hausdorff_dist = (np.max(dist_ref) + np.max(dist_seg)) / 2.0
    # hd_95 = (np.percentile(dist_ref, 95) + np.percentile(dist_seg, 95)) / 2.0
    # HD_rmse = (np.sqrt(np.mean(dist_ref ** 2)) + np.sqrt(np.mean(dist_seg ** 2))) / 2.0

    hd1 = np.max(dist_ref)
    hd2 = np.max(dist_seg)
    hd_95_1 = np.percentile(dist_ref, 95)
    hd_95_2 = np.percentile(dist_seg, 95)
    hd_rmse_1 = np.sqrt(np.mean(np.square(dist_ref)))
    hd_rmse_2 = np.sqrt(np.mean(np.square(dist_seg)))


    with open(os.path.join(root_dir, 'eval_metric_resampled.csv'), 'a') as f:
        f.write(f'{bone},{pt},{cond},{hd1},{hd2},{hd_95_1},{hd_95_2},{hd_rmse_1},{hd_rmse_2}\n')




df = pd.read_csv(os.path.join(root_dir,'eval_metric_resampled.csv'))
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(x='INPUT TYPE', y='HDmax1', data=df, palette='Set3')
plt.title('Max Hausdorff Distance')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.subplot(1, 3, 2)
sns.boxplot(x='INPUT TYPE', y='HD95_1', data=df, palette='Set3')
plt.title('95% Hausdorff Distance')

plt.subplot(1, 3, 3)
sns.boxplot(x='INPUT TYPE', y='HDrmse1', data=df, palette='Set3')
plt.title('RMSE of Hausdorff Distance')

plt.tight_layout()
plt.savefig(os.path.join(root_dir, 'eval_metric_resampled_boxplot.png'))
plt.show()
plt.close()


grouped = df.groupby('INPUT TYPE')[['HDmax1', 'HD95_1', 'HDrmse1']].agg(['mean', 'std'])
print(grouped)
grouped.to_csv(os.path.join(root_dir, 'eval_metric_resampled_grouped.csv'))