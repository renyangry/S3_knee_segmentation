import os
from re import X 
import sys
import glob
import json
from pycpd import RigidRegistration, DeformableRegistration, AffineRegistration
import numpy as np
import trimesh
sys.path.append('/home/rgu/Documents/Mario-s3_DesktopPlannerAlgorithm-master')
from numpy.testing import assert_array_almost_equal

from mario_ssm import SimpleGenSSAM
from mario_vtk import SimpleVtkExtraction
from mario_vtk_fun import read_shape, write_shape
from mario_morph import SimpleReconstruction3D



root_dir = '/home/rgu/Documents'    
d_stls = os.path.join(root_dir,'mesh_ssm', 'simplify_stl')
d_outs = os.path.join(root_dir,'mesh_ssm', 'output')
os.makedirs(d_outs, exist_ok=True)
d_dicom = ""

# ------------------------------------------------------------------------------
# different with mario, mario computes median shaped reference mesh
reference_mesh_path = os.path.join(d_stls,'halvleg_1064.stl')
ref_mesh = trimesh.load(reference_mesh_path)

# test with example 
moving_mesh_path = glob.glob(os.path.join(d_stls, '*.stl'))
mov_mesh = trimesh.load(moving_mesh_path[0])

# actual registration - rigid first?
anchor_vertices = np.asarray(ref_mesh.vertices)
moving_vertices = np.asarray(mov_mesh.vertices)
reg = RigidRegistration(X=anchor_vertices, Y=moving_vertices)
# reg = RigidRegistration(**{'X':anchor_vertices,'Y':moving_vertices})

new_vertices, (scale, rotation_, translation_) = reg.register()
new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=mov_mesh.faces)
# new_vertices = reg.TY
# scale = reg.s
# R = reg.R.tolist()
# t = reg.t.tolist()
rig_dict = {
        'S': scale,
        'R': rotation_,
        'T': translation_,
        }


assert_array_almost_equal(new_vertices, anchor_vertices, decimal=0)


rig_dict['anchor_name'] = os.path.basename(reference_mesh_path)
new_mesh.export(os.path.join(d_outs,os.path.basename(moving_mesh_path[0])))
with open(d_outs + '/rigid.json', 'w') as f:
        json.dump(rig_dict, f)
        

# ------------------------------------------------------------------------------
# affine registration
reg = AffineRegistration(X=anchor_vertices, Y=moving_vertices)
new_vertices, (b_, t_) = reg.register()
new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=mov_mesh.faces)

aff_dict = {
        'B': b_,
        'T': t_,
        }


assert_array_almost_equal(new_vertices, anchor_vertices, decimal=0)


aff_dict['anchor_name'] = os.path.basename(reference_mesh_path)
new_mesh.export(os.path.join(d_outs,os.path.basename(moving_mesh_path[0])))
with open(d_outs + '/affine.json', 'w') as f:
        json.dump(aff_dict, f)