import os
import sys
import numpy as np
import ants
import SimpleITK as sitk
import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#------------------------------------------------------------------------------------------------------------#
out_dir = '/home/rgu/Documents/surgical_planning'
os.makedirs(out_dir, exist_ok=True)
if not os.path.exists(os.path.join(out_dir, 'surgical_plan_coordinates.csv')):
    with open(os.path.join(out_dir, 'surgical_plan_coordinates.csv'), 'w') as f:
        f.write('FILENAME,INPUT TYPE,FEMUR WIDTH,ME_x,ME_y,ME_z,LE_x,LE_y,LE_z,IMPLANT_x,IMPLANT_y,IMPLANT_z\n')
        
#------------------------------------------------------------------------------------------------------------#
bone_path = '/home/rgu/Documents/test/left_femur'
cond = "proximal_only"
for filename in os.listdir((bone_path)):
    
    original_img = nib.load(os.path.join(bone_path, filename))
    original_affine = original_img.affine
    original_header = original_img.header

    #------------------------------------------------------------------------------------------------------------#
    # step 1 - load the SSM reconstructed output, and crop to non-zero region for faster computation
    rec_img = ants.image_read(os.path.join(bone_path, filename))
    rec_img_spacing = rec_img.spacing
    volume_slice = int(150/rec_img_spacing[2])
    rec_left_femur = ants.utils.crop_image(rec_img, label=1)
    rec_lf_seg = rec_left_femur.numpy()

    #------------------------------------------------------------------------------------------------------------#
    # step 2 - crop the distal femur - 150 mm^3 volume + surface map
    distal_femur = rec_lf_seg[:, :, 0:volume_slice]

    #------------------------------------------------------------------------------------------------------------#
    # step 3 - calculate the slice-wise volume of step 2, and plot the volume curve for refinement 
    volume = []
    for i in range(distal_femur.shape[2]):
        volume.append(np.sum(distal_femur[:, :, i]))
    volume = np.array(volume)

    plt.figure()
    plt.title('Volume curve of distal femur')
    plt.xlabel('Slice number')
    plt.ylabel('Volume')
    plt.plot(range(distal_femur.shape[2]), volume)
    volume_mean = np.mean(volume)
    plt.plot(volume_mean, 'x', color='green', label='Mean')
    plt.legend()
    plt.show()

    slice_number = np.where(volume > volume_mean)[0]
    new_distal_femur = distal_femur[:, :, slice_number[0]:slice_number[-1]]
    nib.save(nib.Nifti1Image(new_distal_femur, original_affine, original_header), os.path.join(out_dir, str('distal_')+filename))

    #------------------------------------------------------------------------------------------------------------#
    # step 4 - find the medial and lateral points of the distal femur
    img = ants.image_read(os.path.join(out_dir, str('distal_')+filename))
    # crop to knee region
    cropped_img = ants.utils.crop_image(img, label=1)
    # find the medial and lateral points
    [y,z] = np.where(cropped_img.numpy()[0,:,:] != 0)
    x = np.zeros(y.shape).astype(int)
    medial_points = [[x[i], y[i], z[i]] for i in range(len(x))]

    [y_,z_] = np.where(cropped_img.numpy()[-1,:,:] != 0)
    x_ = np.ones(y_.shape).astype(int) * (cropped_img.numpy().shape[0]-1)
    lateral_points = [[x_[i], y_[i], z_[i]] for i in range(len(x_))]

    # find the distance between each pair of medial and lateral points
    distance = np.zeros((len(medial_points), len(lateral_points)))
    for i in range(len(medial_points)):
        for j in range(len(lateral_points)):
            distance[i,j] = np.sqrt((medial_points[i][0]-lateral_points[j][0])**2 + (medial_points[i][1]-lateral_points[j][1])**2 + (medial_points[i][2]-lateral_points[j][2])**2)

    # only selecting the top 0.5% of the distance values
    distance_99 = np.percentile(distance, 99.5)
    index1,index2 = np.where(distance > distance_99)

    # save the femur width, medial and lateral points
    max_diameters_list = [distance[x,y] for x,y in zip(index1, index2)]
    ME_coordinates = [medial_points[x] for x in index1]
    LE_coordinates = [lateral_points[y] for y in index2]

    # calculate the implant center as the midpoint of the medial and lateral points
    coordinates_int = []
    for i in range(len(ME_coordinates)):
        coordinates_int.append([(ME_coordinates[i][0]+LE_coordinates[i][0])/2, (ME_coordinates[i][1]+LE_coordinates[i][1])/2, (ME_coordinates[i][2]+LE_coordinates[i][2])/2])

    for i in range(len(coordinates_int)):
        with open(os.path.join(out_dir, 'surgical_plan_coordinates.csv'), 'a') as f:
            f.write(f'{filename},{cond},{max_diameters_list[i]},{ME_coordinates[i]},{LE_coordinates[i]},{coordinates_int[i]}\n')

    #------------------------------------------------------------------------------------------------------------#
    # step 5 - plot the implant center on the femur
    data = cropped_img.numpy()
    for i in range(len(coordinates_int)):
        location = [round(idx) for idx in coordinates_int[i]]
        data[tuple(location)] = 5
        data[tuple(ME_coordinates[i])] = 10
        data[tuple(LE_coordinates[i])] = 10
 
    nib.save(nib.Nifti1Image(data, original_affine, original_header), os.path.join(out_dir, str('result_'+filename)))



