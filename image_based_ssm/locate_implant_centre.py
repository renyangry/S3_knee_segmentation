#!~/miniconda3/bin/python3.11

import os
import ants
import numpy as np
from ssm_utils import *
from ssm_config import *
import nibabel as nib
from scipy.ndimage import distance_transform_edt
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


def distance_to_line(point, line_point1, line_point2):
    v = line_point2 - line_point1
    p1_to_p3 = point - line_point1
    distance = np.linalg.norm(np.cross(v, p1_to_p3)) / np.linalg.norm(v)
    return distance


#------------------------------------------------------------------------------------------------------------#
if not os.path.exists(os.path.join(OUT_DIR, 'surgical_plan_coordinates.csv')):
    with open(os.path.join(OUT_DIR, 'surgical_plan_coordinates.csv'), 'w') as f:
        f.write('FILENAME,INPUT TYPE,FEMUR WIDTH,FEMUR HEAD,ME,LE,IMPLANT\n')

bone = "left_femur"
cond = "proximal_only"
# for bone in BONE_STRUCTURE:
        # for cond in CONDITION:
print(f'Processing {bone}...')
print(f'The input data uses {cond}...')

test_dir = TEST_DIRS[bone]
results_dir = RESULTS_DIRS_TESTING[bone][cond]
# for filename in os.listdir((test_dir)):
filename = os.listdir(test_dir)[0]

#------------------------------------------------------------------------------------------------------------#
# step 1 - load the SSM reconstructed output, and crop to non-zero region for faster computation
rec_img = ants.image_read(os.path.join(test_dir, filename))
rec_img_spacing = rec_img.spacing
volume_slice = int(150/rec_img_spacing[2])
rec_left_femur = ants.utils.crop_image(rec_img, label=1)
rec_lf_seg = rec_left_femur.numpy()

#------------------------------------------------------------------------------------------------------------#
# step 2 - locate the femur head center by calculating Euclidean Distance Transform (EDT) per pixel
proximal_femur = rec_lf_seg[:, :, -volume_slice::]
distance_map = distance_transform_edt(proximal_femur)
max_distance = np.max(distance_map)
femur_head_centre = np.unravel_index(np.argmax(distance_map), distance_map.shape)

print(f"Longest Distance: {max_distance}")
print(f"Position of Longest Distance (estimated femur head center): {femur_head_centre}")

#------------------------------------------------------------------------------------------------------------#
# step 3 - find the longest femur diameter at distal end
distal_femur = rec_lf_seg[:, :, 0:volume_slice]
sitk_image = sitk.GetImageFromArray(distal_femur)
sitk_image_int = sitk.Cast(sitk_image, sitk.sitkUInt8)  # Adjust the target pixel type if needed
reference_surface = sitk.LabelContour(sitk_image_int)
surface_array = sitk.GetArrayFromImage(reference_surface)

max_diameter = 0
max_diameter_slice = 0
for slice_number in range(surface_array.shape[2]):
    current_slice = surface_array[:, :, slice_number]
    coords = np.array(np.nonzero(current_slice)).T
    distances = squareform(pdist(coords))
    i, j = np.unravel_index(np.argmax(distances), distances.shape)
    coordinate1 = tuple(coords[i]) + (slice_number,)
    coordinate2 = tuple(coords[j]) + (slice_number,)
    diameter = distances[i, j]

    if diameter > max_diameter:
        max_diameter = diameter
        max_diameter_slice = slice_number
        max_diameter_coordinates = [coordinate1, coordinate2]

print("Max Femur Diameter:", max_diameter)
print("Slice Number:", max_diameter_slice)
print("Coordinates:", max_diameter_coordinates)

#------------------------------------------------------------------------------------------------------------#
# step 4 - find the minimal distance from femur head center to femur width (|LE-ME|)
fh_centre_coordinate = np.array(femur_head_centre[0:2]+(rec_lf_seg.shape[2]-proximal_femur.shape[2]+femur_head_centre[2],))
point1 = np.array(max_diameter_coordinates[0])
point2 = np.array(max_diameter_coordinates[1])

distance = distance_to_line(fh_centre_coordinate, point1, point2)
t_value = np.dot(fh_centre_coordinate - point1, point2 - point1) / np.dot(point2 - point1, point2 - point1)
shortest_distance_coordinates = point1 + t_value * (point2 - point1)

print(f"The shortest distance from femur head center to the line is: {distance}")
print(f"The  absolute coordinates that give the shortest distance are: {shortest_distance_coordinates.astype('int')}")

#------------------------------------------------------------------------------------------------------------#
# step 5 - transform all coordinate back to the full leg volume
implant_seg = rec_lf_seg.copy()
proximal_femur[femur_head_centre] = 5
implant_seg[:, :, -volume_slice::] = proximal_femur

distal_femur[max_diameter_coordinates[0]] = 5
distal_femur[max_diameter_coordinates[1]] = 5
coordinates_int = shortest_distance_coordinates.astype('int')
# x_range = slice(coordinates_int[0] - 3, coordinates_int[0] + 4)
# y_range = slice(coordinates_int[1] - 3, coordinates_int[1] + 4)
# distal_femur[x_range, y_range, coordinates_int[2]] = 3
distal_femur[tuple(coordinates_int)] = 3
implant_seg[:, :, 0:volume_slice] = distal_femur

# femur_volume = ants.from_numpy(implant_seg)
# outs = ants.registration(rec_img, femur_volume, type_of_transforme='Similarity')
# ants.image_write(outs, os.path.join('/home/rgu/Documents/test/11.nii.gz'))
# indices = np.where(outs.numpy() == 3)
# print("Coordinates of landmarks:")
# for coordinate in indices:
#     print(tuple(coordinate))

#------------------------------------------------------------------------------------------------------------#
# save these landmark coordinates in file
with open(os.path.join(OUT_DIR, 'surgical_plan_coordinates.csv'), 'a') as f:
    f.write(f'{filename},{cond},{max_diameter},{str(femur_head_centre)},{str(max_diameter_coordinates[0])},{str(max_diameter_coordinates[1])},{str(tuple(coordinates_int))}\n')

#------------------------------------------------------------------------------------------------------------#
# save the output
rotated_image = np.rot90(distal_femur[:, :, max_diameter_slice], k=1)
plt.imshow(rotated_image, cmap='gray')
plt.scatter(point1[0], rotated_image.shape[0] - point1[1], color='red', marker='x', label='Medial Epicondyle')
plt.scatter(point2[0], rotated_image.shape[0] - point2[1], color='blue', marker='x', label='Lateral Epicondyle')
plt.scatter(shortest_distance_coordinates[0], rotated_image.shape[0] - shortest_distance_coordinates[1], color='green', marker='x', label='Implant Center')
plt.plot([point1[0], point2[0]], [rotated_image.shape[0] - point1[1], rotated_image.shape[0] - point2[1]], color='yellow', linestyle='--', label='Femur Width')
plt.legend(loc='upper right')
plt.savefig(os.path.join(OUT_DIR, filename + '.png'), bbox_inches='tight')
plt.show()
