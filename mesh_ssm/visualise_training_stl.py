from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import os
import glob



root_dir = '/home/rgu/Documents'
out_path = glob.glob(os.path.join(root_dir, 'mesh_ssm', 'mesh', '*.stl'))
figure = pyplot.figure()
axes = figure.add_subplot(projection='3d')
# 1045 1115 tibia
your_mesh = trimesh.load(out_path[0])
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vertices[your_mesh.faces]))
# Auto scale to the mesh size
scale = your_mesh.vertices.flatten()
axes.auto_scale_xyz(scale, scale, scale)
axes.set_xlabel('X')
axes.set_ylabel('Y')
axes.set_zlabel('Z')
# Show the plot to the screen
pyplot.show()