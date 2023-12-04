import os 
import ants
import json
import glob
import nibabel as nib
import nibabel.processing as nibproc
from skimage.measure import marching_cubes#, marching_cubes_lewiner
import numpy as np
from stl import mesh
from pycpd import RigidRegistration, AffineRegistration, DeformableRegistration
import trimesh



def save_training_json(moving_train_dir, original_train_dir, json_path):
    left_femur_list = []
    for filename in os.listdir(moving_train_dir):
        left_femur_list.append(os.path.join(original_train_dir, os.path.basename(filename)))
    
    with open(json_path, 'w') as f:
        json.dump({'ssm_list': left_femur_list}, f)


def extract_femur(im, label):
    # im = ants.image_read(path)
    mask_img = ants.utils.get_mask(im, label, label+1)
    cropped = ants.utils.crop_image(mask_img, mask_img)
    resampled = ants.resample_image(cropped, [1,1,1])
    # smoothed = ants.utils.smooth_image(resampled, 1.5)
    padded = ants.utils.pad_image(resampled, pad_width=[(20, 20), (20, 20), (20, 20)])
    return padded


def convert_binary2mesh(im, filename, out_path):
    os.makedirs(out_path, exist_ok=True)

    verts, faces, normals, values = marching_cubes(im, 0)
    # for i, f in enumerate(faces):
    #     for j in range(faces.shape[1]):
    #         obj_3d.vectors[i][j] = verts[f[j]]
    
    obj_3d = trimesh.Trimesh(vertices=verts, faces = faces)

    mesh_name = os.path.basename(filename).split('.')[0] + '.stl'
    obj_3d.export(os.path.join(out_path, mesh_name))   
    
    
def simplify_mesh(in_path, out_path, bin_path, factor):
# Fast-Quadric-Mesh Simplication
    mesh_path = glob.glob(os.path.join(in_path, '*.stl'))
    os.makedirs(out_path, exist_ok=True)

    for stl_filename in mesh_path: 
        cur_filename = os.path.basename(stl_filename)
        out_file = os.path.join(out_path, cur_filename)
        os.system(bin_path + ' ' + stl_filename + ' ' + out_file + ' ' + str(factor))
    
    
def scale_mesh(mesh, factor=.1):

    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)
    vertices *= factor
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

def rigid_reg_meshes(ref_mesh, mov_mesh):
    # input need to be coordinates of vertices
    anchor_vertices = np.asarray(ref_mesh.vertices)
    moving_vertices = np.asarray(mov_mesh.vertices)
    reg = RigidRegistration(X=anchor_vertices, Y=moving_vertices)
    reg.register()

    new_vertices = reg.TY
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=mov_mesh.faces)
    scale = reg.s
    rotation_ = reg.R.tolist()
    translation_ = reg.t.tolist()
    trans_dict = {
            's': scale,
            'R': rotation_,
            't': translation_,
            }
    return new_mesh, trans_dict


def affine_reg_meshes(ref_mesh, mov_mesh):
    # input need to be coordinates of vertices
    anchor_vertices = np.asarray(ref_mesh.vertices)
    moving_vertices = np.asarray(mov_mesh.vertices)
    reg = AffineRegistration(X=anchor_vertices, Y=moving_vertices)
    reg.register()

    new_vertices = reg.TY
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=mov_mesh.faces)
    B_ = reg.B
    translation_ = reg.t.tolist()
    trans_dict = {
            'B': B_,
            't': translation_,
            }
    return new_mesh, trans_dict


def deformable_reg_meshes(ref_mesh, mov_mesh):
    # input need to be coordinates of vertices
    anchor_vertices = np.asarray(ref_mesh.vertices)
    moving_vertices = np.asarray(mov_mesh.vertices)
    reg = DeformableRegistration(X=anchor_vertices, Y=moving_vertices)
    reg.register()

    new_vertices = reg.TY
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=mov_mesh.faces)
    B_ = reg.B
    translation_ = reg.t.tolist()
    trans_dict = {
            'B': B_,
            't': translation_,
            }
    return new_mesh, trans_dict

# scaling mesh vertices