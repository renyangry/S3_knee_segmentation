import os 
import ants
import json
import nibabel as nib
import nibabel.processing as nibproc
from skimage.measure import marching_cubes #marching_cubes_lewiner
import numpy as np
from stl import mesh


def save_training_json(moving_train_dir, original_train_dir, json_path):
    left_femur_list = []
    for filename in os.listdir(moving_train_dir):
        left_femur_list.append(os.path.join(original_train_dir, os.path.basename(filename)))
    
    with open(json_path, 'w') as f:
        json.dump({'ssm_list': left_femur_list}, f)


# def interpolate_all_masks(JSON_PATH, out_path): 
#     with open(JSON_PATH, 'r') as f:
#         paths = json.load(f)
#     os.makedirs(out_path, exist_ok=True)
#     for cur_path in paths["ssm_list"]:
#         print(cur_path)
#         cur_filename = os.path.basename(cur_path)
#         cur_im = nib.load(cur_path)
#         resampled_im = nibproc.resample_to_output(cur_im, (1,1,1))
#         nib.save(resampled_im, os.path.join(out_path, cur_filename))


def extract_femur(im, label):
    # im = ants.image_read(path)
    mask_img = ants.utils.get_mask(im,label,label+1)
    cropped = ants.utils.crop_image(mask_img, mask_img)
    resampled = ants.resample_image(cropped, [1,1,1])
    padded = ants.utils.pad_image(resampled,pad_width=[(20, 20), (20, 20), (20, 20)])
    return padded


def convert_binary2mesh(im, filename, out_path):
    os.makedirs(out_path, exist_ok=True)

    # smoothed = mcubes.smooth(im, method='gaussian')
    # vertices, triangles, _, _ = marching_cubes_lewiner(im, 0, allow_degenerate=False)
    # because sitk images have z direction first, we must swap x and z coordinates
    # vertices = vertices[:, ::-1]
    # convert pixel coordinates to world coordinates using the image's affine
    # vertices = np.transpose(vertices)
    # ones = np.ones((1, vertices.shape[1]))
    # vertices = np.concatenate([vertices, ones])
    # vertices = np.matmul(affine, vertices)
    # vertices = vertices[:3,:]
    # vertices = np.transpose(vertices)
    # # save the resulting set of vertices and triangles
    # mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    # mesh_name = os.path.basename(filename).split('.')[0] + '.stl'
    # mesh.export(os.path.join(out_path, mesh_name))

    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # # Fancy indexing: `verts[faces]` to generate a collection of triangles
    # mesh = Poly3DCollection(vertices[triangles])
    # mesh.set_edgecolor('k')
    # ax.add_collection3d(mesh)
    # ax.set_xlabel("x-axis: a = 6 per ellipsoid")
    # ax.set_ylabel("y-axis: b = 10")
    # ax.set_zlabel("z-axis: c = 16")
    # ax.set_xlim(0, 24)  # a = 6 (times two for 2nd ellipsoid)
    # ax.set_ylim(0, 20)  # b = 10
    # ax.set_zlim(0, 32)  # c = 16

    # plt.tight_layout()
    # plt.show()

    verts, faces, normals, values = marching_cubes(im, 0)
    obj_3d = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

    for i, f in enumerate(faces):
        obj_3d.vectors[i] = verts[f]
    mesh_name = os.path.basename(filename).split('.')[0] + '.stl'
    obj_3d.save(os.path.join(out_path, mesh_name))