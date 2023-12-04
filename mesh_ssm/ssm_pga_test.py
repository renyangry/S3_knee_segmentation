import glob
import os
import pyvista as pv
import numpy as np
from morphomatics.geom import Surface
from morphomatics.stats import StatisticalShapeModel
from morphomatics.manifold import FundamentalCoords


# load surfaces
root_dir = '/home/rgu/Documents'
d_stls = os.path.join(root_dir,'mesh_ssm','mesh')
moving_mesh_path = glob.glob(os.path.join(d_stls, '*.stl'))
meshes = [pv.read(moving_mesh_path[i]) for i in range(len(moving_mesh_path))]

# show
pl = pv.Plotter(notebook=True, shape=(1,3))
for i in range(3):
    pl.subplot(0, i)
    pl.add_mesh(meshes[i])
    # pl.view_yx()
    # pl.camera.roll += 180
    pl.camera.zoom(2)
pl.show(jupyter_backend='static')


# to Surface type
as_surface = lambda mesh: Surface(mesh.points, mesh.faces.reshape(-1, 4)[:, 1:])
surfaces = [as_surface(m) for m in meshes]

# construct model
SSM = StatisticalShapeModel(lambda ref: FundamentalCoords(ref))
SSM.construct(surfaces)

pl = pv.Plotter(notebook=True)
pl.add_mesh(pv.PolyData(SSM.mean.v, meshes[0].faces))
pl.camera.zoom(2)
pl.show(jupyter_backend='static')


# standard deviation associated to 1st mode
std = np.sqrt(SSM.variances[0])

pl = pv.Plotter(notebook=True, shape=(1,4))
for i, t in enumerate(np.linspace(-1.0,1.0,4)):
    # excite mode
    coords = SSM.space.exp(SSM.mean_coords, t * std * SSM.modes[0])
    # map shape space coords to vertex coords
    v = SSM.space.from_coords(coords)
    # add mesh to plot
    pl.subplot(0, i)
    pl.add_mesh(pv.PolyData(v, meshes[0].faces))
    pl.camera.zoom(2)
pl.show(jupyter_backend='static')
