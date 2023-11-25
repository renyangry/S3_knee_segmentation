from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import os
import glob

out_path = glob.glob(os.path.join(root_dir, 'mesh_ssm','mesh'), '*.stl')
# Create a new plot
figure = pyplot.figure()
axes = figure.add_subplot(projection='3d')

# Load the STL files and add the vectors to the plot
your_mesh = mesh.Mesh.from_file(out_path[0])
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

# Auto scale to the mesh size
scale = your_mesh.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)

# Show the plot to the screen
pyplot.show()