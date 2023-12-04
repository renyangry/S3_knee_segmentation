import pyvista as pv
import numpy as np
import nibabel as nib

# Load the volumes (use your actual data loading method here)
vol1 = pv.UniformGrid("")
vol2 = pv.UniformGrid("")


# Load the volumes
vol1 = nib.load('/home/rgu/Documents/test/left_femur/femur_left_BR-119BR-119_dicom.nii.gz')
vol2 = nib.load('/home/rgu/Documents/ssm_results/left_femur/proximal_only/femur_left_BR-119BR-119_dicom.nii.gz')

# Extract the volume data
data1 = vol1.get_fdata()
data2 = vol2.get_fdata()

# Create PyVista UniformGrids from the volume data
grid1 = pv.UniformGrid()
grid1.dimensions = data1.shape
grid1.origin = vol1.affine[:3, 3]
grid1.spacing = vol1.affine.diagonal()[:3]
grid1.point_arrays['data'] = data1.flatten(order='F')

grid2 = pv.UniformGrid()
grid2.dimensions = data2.shape
grid2.origin = vol2.affine[:3, 3]
grid2.spacing = vol2.affine.diagonal()[:3]
grid2.point_arrays['data'] = data2.flatten(order='F')

# Create a plotting window
plotter = pv.Plotter(notebook=True)

# Add volumes to the plotter
plotter.add_volume(grid1, cmap='Blues', opacity=0.5)
plotter.add_volume(grid2, cmap='Reds', opacity=0.5)

# Adjust settings, view, and display
plotter.camera_position = 'xy'
plotter.show(auto_close=False)


# Capture the screenshots and save as images
plotter.orbit_on_path(n_points=36)  # Rotate around the object
plotter.screenshot("output_image_%05d.png", transparent_background=True)  # Save screenshots

# Close the plotter window
plotter.close()

# Use `ffmpeg` or similar tools to convert the images to a video
# ffmpeg -r 20 -i output_image_%05d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p output_video.mp4
