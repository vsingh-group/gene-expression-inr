# %%
import numpy as np
import nibabel as nib
from modules import vox2mni, mni2vox

file = "MNI152_T1_1mm_brain"
atlas = nib.load(f"./data/{file}.nii.gz")
atlas_data = atlas.get_fdata()
affine = atlas.affine

regions = np.unique(atlas_data)
label_voxel_coordinates = {label: np.argwhere(atlas_data == label) for label in regions}

# voxel_coord = (x, y, z)
# region = atlas_data[voxel_coord]


# %%
