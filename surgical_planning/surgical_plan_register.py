import os
import sys
import numpy as np
import ants
import SimpleITK as sitk
import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import distance_transform_edt
import pandas as pd
from scipy.ndimage import label, center_of_mass, binary_erosion
#------------------------------------------------------------------------------------------------------------#
out_dir = '/home/rgu/Documents/surgical_planning_registration'
os.makedirs(out_dir, exist_ok=True)
if not os.path.exists(os.path.join(out_dir, 'surgical_plan_registration.csv')):
    with open(os.path.join(out_dir, 'surgical_plan_coordinates.csv'), 'w') as f:
        f.write('FILENAME,INPUT TYPE,ME_x,ME_y,ME_z,LE_x,LE_y,LE_z,rIMPLANT_x,rIMPLANT_y,rIMPLANT_z,mIMPLANT_x,mIMPLANT_y,mIMPLANT_z \n')
if not os.path.exists(os.path.join(out_dir, 'surgical_plan_reg2.csv')):
    with open(os.path.join(out_dir, 'surgical_plan_reg2.csv'), 'w') as f:
        f.write('FILENAME,INPUT TYPE,implant_coord \n')

#------------------------------------------------------------------------------------------------------------#
# step 1- create a mask for ME, LE and centre of knee implant
bone_path = '/home/rgu/Documents/test/left_femur'
cond = "proximal_only"
ref_path = os.path.join(bone_path, 'femur_left_BR-002BR-002_dicom.nii.gz')
ref_img = ants.image_read(ref_path)
landmark_mask = np.zeros(ref_img.numpy().shape)
ref_ME = [37, 184, 127] # [37, 187, 129], already -1 for python indexing
ref_LE = [119, 224, 127]
medial_point = [(ref_ME[0]+ref_LE[0])//2, (ref_ME[1]+ref_LE[1])//2, (ref_ME[2]+ref_LE[2])//2]
medial_point = [int(coord) for coord in medial_point]

landmark_mask[tuple(ref_ME)] = 1.0
landmark_mask[tuple(ref_LE)] = 1.0
landmark_mask[tuple(medial_point)] = 1.0
landmark_mask = ants.from_numpy(landmark_mask, origin=ref_img.origin, spacing=ref_img.spacing, direction=ref_img.direction)
dilated_mask = ants.utils.morphology(landmark_mask, operation='dilate', radius=5)
# ants.image_write(dilated_mask, os.path.join(out_dir, 'dilated_mask.nii.gz'))
#ants.image_write(landmark_mask, os.path.join(out_dir, 'landmark_mask.nii.gz'))

#------------------------------------------------------------------------------------------------------------#
# step 2 - data preparation, transform the pixel coordinates to physical coordinates
ME_pp = ants.transform_index_to_physical_point(ref_img, tuple(ref_ME))
LE_pp = ants.transform_index_to_physical_point(ref_img, tuple(ref_LE))
medial_pp = ants.transform_index_to_physical_point(ref_img, tuple(medial_point))
# print(f"Physical Point of ME: {ME_pp}")
# print(f"Physical Point of LE: {LE_pp}")
# print(f"Physical Point of Medial: {medial_pp}")

d = {'x': [ME_pp[0], medial_pp[0], LE_pp[0]], 'y': [ME_pp[1], medial_pp[1], LE_pp[1]], 'z': [ME_pp[2], medial_pp[2], LE_pp[2]]}
pts = pd.DataFrame(data=d)
# print("Physical Point in Reference Image:")
# print(pts)

# d = ants.utils.label_stats(landmark_mask, landmark_mask)
# print(f"{d}")

#------------------------------------------------------------------------------------------------------------#
# step 3 - image registration and transformation matrix calculation
for filename in os.listdir((bone_path)):
    mov_img = ants.image_read(os.path.join(bone_path, filename))
    mov_femur = ants.crop_image(mov_img, label=1)
    # ants.image_write(mov_femur, os.path.join(out_dir, filename))

    outs = ants.registration(mov_femur, ref_img, 'Elastic') #Elastic
    warped_img = outs['warpedfixout']
    new_mask = ants.apply_transforms(mov_femur, dilated_mask, transformlist=outs['fwdtransforms'])
    new_mask[new_mask >= 0.5] = 1
    new_mask[new_mask < 0.5] = 0
    # ants.image_write(new_mask, os.path.join(out_dir, str('seg_')+filename))

    # transformed_pointa = ants.apply_transforms_to_points(3, d, transformlist=outs['fwdtransforms'], verbose=True)
    # print("Transformed Point in Testing Image:")
    # print(transformed_pointa)
 
    transformed_point = ants.apply_transforms_to_points(3, pts, transformlist=outs['fwdtransforms'], whichtoinvert=[0,1], verbose=True)
    # print("Transformed Point in Testing Image:")
    # print(transformed_point)
    ME_index = ants.transform_physical_point_to_index(mov_femur, tuple([transformed_point['x'][0], transformed_point['y'][0], transformed_point['z'][0]]))
    ME_index = [int(coord)+1 for coord in ME_index]
    print(f"Physical Point of ME: {ME_index}")
    medial_index = ants.transform_physical_point_to_index(mov_femur, tuple([transformed_point['x'][1], transformed_point['y'][1], transformed_point['z'][1]]))
    medial_index = [int(coord)+1 for coord in medial_index]
    print(f"Physical Point of Medial: {medial_index}")
    LE_index = ants.transform_physical_point_to_index(mov_femur, tuple([transformed_point['x'][2], transformed_point['y'][2], transformed_point['z'][2]]))
    LE_index = [int(coord)+1 for coord in LE_index]
    print(f"Physical Point of LE: {LE_index}")
    measured_coord = [(ME_index[0]+LE_index[0])//2, (ME_index[1]+LE_index[1])//2, (ME_index[2]+LE_index[2])//2]
    measured_coord = [int(coord) for coord in measured_coord]
    print(f"Physical Point of Measured Medial: {measured_coord}")
#------------------------------------------------------------------------------------------------------------#
# step 4 - save the registered landmark coordinates
    with open(os.path.join(out_dir, 'surgical_plan_coordinates.csv'), 'a') as f:
        f.write(f'{filename},{cond},{ME_index[0]},{ME_index[1]},{ME_index[2]},{LE_index[0]},{LE_index[1]},{LE_index[2]},{medial_index[0]},{medial_index[1]},{medial_index[2]},{measured_coord[0]},{measured_coord[1]},{measured_coord[2]}\n')

#------------------------------------------------------------------------------------------------------------#
# step 5 - for comparison only, alternative method to calculate the implant center  
    # ---------------test ----------
    new_mask = new_mask.numpy()
    output = new_mask * mov_femur.numpy()
    # ants.image_write(ants.from_numpy(output, origin=mov_femur.origin, spacing=mov_femur.spacing, direction=mov_femur.direction), os.path.join(out_dir, str('out_')+filename))
    
    labeled_clusters, num_clusters = label(output)
    structuring_element = np.ones((4,4,4), dtype=np.uint8)
    eroded_label_map = np.zeros_like(output)

    for cluster_num in range(num_clusters):
        cluster_mask = (labeled_clusters == cluster_num)
        eroded_cluster = binary_erosion(cluster_mask, structure=structuring_element)
        eroded_label_map[eroded_cluster] = cluster_num

    coordinates = np.where(eroded_label_map > 0)
    print("Coordinates where Eroded Label Map > 0:")
    for coord in zip(*coordinates):
        coord = [int(c)+1 for c in coord]
        print(coord)
        with open(os.path.join(out_dir, 'surgical_plan_reg2.csv'), 'a') as f:
            f.write(f'{filename},{cond},{coord}\n')


    # ---------------test ----------
    # break
