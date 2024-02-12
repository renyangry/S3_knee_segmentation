from math import e
import os
from re import X 
import sys
import glob
import json
from turtle import st
from pycpd import RigidRegistration, DeformableRegistration, AffineRegistration
import numpy as np
import trimesh
sys.path.append('/home/rgu/Documents/Mario-s3_DesktopPlannerAlgorithm-master')
from numpy.testing import assert_array_almost_equal
import time

root_dir = '/home/rgu/Documents'    
d_stls = os.path.join(root_dir,'new_ssm', 'simplify_stl')
d_outs = os.path.join(root_dir,'new_ssm', 'output')
os.makedirs(d_outs, exist_ok=True)
d_dicom = ""

# ------------------------------------------------------------------------------
# different with mario, mario computes median shaped reference mesh
reference_mesh_path = os.path.join(d_stls,'halvleg_1025.stl')
ref_mesh = trimesh.load(reference_mesh_path)

# test with example 
moving_mesh_path = glob.glob(os.path.join(d_stls, '*.stl'))
mov_mesh = trimesh.load(moving_mesh_path[0])

anchor_vertices = np.asarray(ref_mesh.vertices)
moving_vertices = np.asarray(mov_mesh.vertices)

# ------------------------------------------------------------------------------
### actual registration - rigid first?
### proven that with orientation correction in preprocessing step, rigid registration is not necessary
# start = time.time()
# reg = RigidRegistration(X=anchor_vertices, Y=moving_vertices)
# end = time.time()
# elapsed = end - start
# print('rigid registration time: ', elapsed) # around 17 seconds

# new_vertices = reg.TY
# ### new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=mov_mesh.faces)
# scale = reg.s
# rotation_ = reg.R.tolist()
# translation_ = reg.t.tolist()
# rig_dict = {
#         'mesh': os.path.basename(moving_mesh_path[0]),
#         'S': scale,
#         'R': rotation_,
#         'T': translation_,
#         }
# # print(rig_dict)
# ### assert_array_almost_equal(new_vertices, anchor_vertices, decimal=0) #Arrays are not almost equal to 0 decimals
# with open(d_outs + '/rigid.json', 'w') as f:
#         json.dump(rig_dict, f)


# ------------------------------------------------------------------------------
# affine registration
start = time.time()
reg = AffineRegistration(X=anchor_vertices, Y=moving_vertices)
end = time.time()
elapsed = end - start
print('affine registration time: ', elapsed)

affine_vertices = reg.TY
new_mesh = trimesh.Trimesh(vertices=affine_vertices, faces=mov_mesh.faces)
b_ = reg.B.tolist()
t_ = reg.t.tolist()

aff_dict = {
        'mesh': os.path.basename(moving_mesh_path[0]),
        'B': b_,
        'T': t_,
        }
print(aff_dict)

# assert_array_almost_equal(new_vertices, anchor_vertices)

new_mesh.export(os.path.join(d_outs,os.path.basename(moving_mesh_path[0])))
with open(d_outs + '/affine.json', 'w') as f:
        json.dump(aff_dict, f)