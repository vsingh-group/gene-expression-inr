# %%
from nilearn.datasets import (
    fetch_atlas_surf_destrieux,
    load_fsaverage,
    load_fsaverage_data,
)
from nilearn.surface import SurfaceImage
from nilearn.plotting import plot_surf_roi, show
import os
import numpy as np
import nibabel as nib
import seaborn as sns
from nilearn import plotting
import matplotlib.pyplot as plt



# Set FreeSurfer environment variables
os.environ['FREESURFER_HOME'] = '/home/xizheng/freesurfer'
os.environ['SUBJECTS_DIR'] = '/home/xizheng/freesurfer/subjects'

# Read the surface data
subject_id = "fsaverage"
hemi = "lh"

# Read annotation
aparc_file = os.path.join(os.environ["SUBJECTS_DIR"],
                         subject_id, "label",
                         hemi + ".aparc.annot")
labels, ctab, names = nib.freesurfer.read_annot(aparc_file)

# Generate random data
rs = np.random.RandomState(4)
roi_data = rs.uniform(.5, .8, size=len(names))

# Map to vertices
vtx_data = roi_data[labels]
vtx_data[labels == -1] = -1

# Get surface geometry
surf_file = os.path.join(os.environ["SUBJECTS_DIR"],
                        subject_id, "surf",
                        hemi + ".inflated")
vertices, faces = nib.freesurfer.read_geometry(surf_file)

# Create the visualization
fig = plt.figure(figsize=(10, 10))
plotting.plot_surf_stat_map(
    surf_mesh=[vertices, faces],
    stat_map=vtx_data,
    hemi='left',
    view='lateral',
    bg_map=None,
    colorbar=True,
    threshold=0,
    cmap=sns.color_palette("rocket", as_cmap=True),
    vmin=0.5,  # Match PySurfer range
    vmax=0.75,  # Match PySurfer range
    alpha=.8
)

##second part


fsaverage = load_fsaverage("fsaverage5")
destrieux = fetch_atlas_surf_destrieux()
destrieux_atlas = SurfaceImage(
    mesh=fsaverage["pial"],
    data={
        "left": destrieux["map_left"],
        "right": destrieux["map_right"],
    },
)

# The labels are stored as bytes for the Destrieux atlas.
# For convenience we decode them to string.
labels = [x.decode("utf-8") for x in destrieux.labels]

# Retrieve fsaverage5 surface dataset for the plotting background.
# It contains the surface template as pial and inflated version.
fsaverage_meshes = load_fsaverage()

print(labels)

# The fsaverage data contains file names pointing to the file locations
# The sulcal depth maps will be is used for shading.
fsaverage_sulcal = load_fsaverage_data(data_type="sulcal")
print(f"{fsaverage_sulcal=}")


plot_surf_roi(
    surf_mesh=fsaverage_meshes["inflated"],
    roi_map=destrieux_atlas,
    hemi="left",
    view="lateral",
    bg_map=fsaverage_sulcal,
    bg_on_data=True,
    darkness=0.5,
)

show()