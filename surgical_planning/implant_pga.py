import os
import ants
import numpy as np
from ssm_utils import *
from ssm_config import *
import nibabel as nib
from skimage.measure import marching_cubes
from stl import mesh
import trimesh
import pyvista as pv
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# def binary2mesh(im):
#     verts, faces, normals, values = marching_cubes(im, 0)
#     obj_3d = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
#
#     for i, f in enumerate(faces):
#         for j in range(faces.shape[1]):
#             obj_3d.vectors[i][j] = verts[f[j]]
#     return obj_3d
    
            
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
rec_img = ants.image_read(os.path.join(results_dir, filename))
rec_img_spacing = rec_img.spacing
volume_slice = int(150/rec_img_spacing[2])
rec_left_femur = ants.utils.crop_image(rec_img, label=1)
rec_lf_seg = rec_left_femur.numpy()

#------------------------------------------------------------------------------------------------------------#
# step 2 - find the longest femur diameter at distal end
distal_femur = rec_lf_seg[:, :, 0:volume_slice]
verts, faces, normals, values = marching_cubes(distal_femur, 0)
mesh = trimesh.Trimesh(vertices=verts, faces=faces)

adjacency_matrix = mesh.vertex_adjacency.toarray()
geodesic_distances = trimesh.graph.shortest_path(adjacency_matrix, return_tree=False)
# center = np.mean(verts, axis=0)
# distances = np.linalg.norm(verts - center, axis=1)

max_inner_diameter = geodesic_distances.max()
max_dist_index = np.unravel_index(np.argmax(geodesic_distances), geodesic_distances.shape)
vertex1 = verts[max_dist_index[0]]
vertex2 = verts.vertices[max_dist_index[1]]

print(f"Maximal inner diameter: {max_inner_diameter}")
print(f"Coordinates of the vertices: Vertex 1: {vertex1}, Vertex 2: {vertex2}")

pl = pv.Plotter()
pl.add_mesh(mesh)
# pl.add_mesh(pv.PolyData(verts, vertex1, vertex2), color='red')
# pl.camera.zoom(2)
pl.show()
