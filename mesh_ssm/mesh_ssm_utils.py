from cgi import test
import os 
import ants
import json
import glob
import nibabel as nib
import nibabel.processing as nibproc
from skimage.measure import marching_cubes #, marching_cubes_lewiner
import numpy as np
from pycpd import RigidRegistration, AffineRegistration, DeformableRegistration
import trimesh
import pyvista as pv


def save_training_json(moving_train_dir, original_train_dir, original_test_dir, json_path):
    train_list = []
    test_list = []
    for filename in os.listdir(moving_train_dir):
        train_list.append(os.path.join(original_train_dir, filename))
    for filenameT in os.listdir(original_test_dir):
        test_list.append(os.path.join(original_test_dir, filenameT))
    
    with open(json_path, 'w') as f:
        json.dump({'training_sample': train_list, 'testing_sample': test_list}, f)


def extract_femur(im, label):
    mask_img = ants.utils.get_mask(im, label, label+1)
    cropped = ants.utils.crop_image(mask_img, mask_img)
    resampled = ants.resample_image(cropped, [1,1,1])
    padded = ants.utils.pad_image(resampled, pad_width=[(20, 20), (20, 20), (20, 20)])
    return padded


def convert_binary2mesh(im, filename, out_path):
    os.makedirs(out_path, exist_ok=True)
    verts, faces, normals, values = marching_cubes(im, 0)
    obj_3d = trimesh.Trimesh(verts, faces)
    mesh_name = os.path.basename(filename).split('.')[0] + '.obj'
    # because fast quadric mesh simplication only takes .obj files \'.stl'
    obj_3d.export(os.path.join(out_path, mesh_name))   
    
    
def simplify_mesh(in_path, out_path, bin_path, factor):
# Fast-Quadric-Mesh Simplication installation needed
# https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification
    mesh_path = glob.glob(os.path.join(in_path, '*.obj'))
    os.makedirs(out_path, exist_ok=True)
    for stl_filename in mesh_path: 
        cur_filename = os.path.basename(stl_filename)
        out_file = os.path.join(out_path, cur_filename)
        os.system(bin_path + ' ' + stl_filename + ' ' + out_file + ' ' + str(factor))


def rename_mesh4ssm(in_path, out_path):
    obj_path = glob.glob(os.path.join(in_path, '*.obj'))
    os.makedirs(out_path, exist_ok=True)
    for obj_filename in obj_path:
        mesh = trimesh.load(obj_filename)
        out_filename = os.path.join(out_path, os.path.basename(obj_filename).split('.')[0]+'.stl')
        mesh.export(out_filename)
        # os.system('mv ' + obj_filename + ' ' + out_filename)


def add_surface_remesh(f_in, clusters=20000, subdivide=3, f_out=None, image_plot=False):
    shape = pv.read(f_in)
    # shape.fill_holes(1000)
    # plot original mesh
    # cow.plot(show_edges=True, color='w')
    clus = pv.pyacvd.Clustering(shape)
    # mesh is not dense enough for uniform remeshing
    clus.subdivide(subdivide)
    clus.cluster(clusters)
    # remesh
    remesh = clus.create_mesh()
    if f_out:
        pv.save_meshio(f_out, remesh)
    # plot uniformly remeshed cow
    if image_plot:
        remesh.plot(color='w', show_edges=True)


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

