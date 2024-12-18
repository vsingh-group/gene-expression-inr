# %%
# nifti_path = 'result_ibf_2full+mirror_TT/nii_9861_83_new/ABCA1_83_new_inrs_avg.nii.gz'
# nifti_path = "result_ibf_2full+mirror_TT/nii_abagen/ABCA1_83_new_abagen.nii.gz"
# my_nifti = nib.load(nifti_path)
# # my_nifti = load_sample_motor_activation_image()

# fsaverage_meshes = load_fsaverage()

# surface_image = SurfaceImage.from_volume(
#     mesh=fsaverage_meshes["pial"],
#     volume_img=my_nifti,
#     interpolation='nearest',
#     radius=0.0,  # Reduce sampling radius
# )

# curv_sign = load_fsaverage_data(data_type="curvature")
# for hemi, data in curv_sign.data.parts.items():
#     print(hemi, data)
#     curv_sign.data.parts[hemi] = np.sign(data)

# fig = plot_surf_stat_map(
#     stat_map=surface_image,
#     surf_mesh=fsaverage_meshes["inflated"],
#     hemi="left",
#     title="Surface with matplotlib",
#     colorbar=True,
#     threshold=0.01,
#     # bg_map=curv_sign,
# )
# fig.show()

# export QT_QPA_PLATFORM=offscreen

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
                         hemi + ".aparc.a2009s.annot")
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

plt.savefig('brain_viz.png', dpi=300, bbox_inches='tight')
plt.close()
# %%
