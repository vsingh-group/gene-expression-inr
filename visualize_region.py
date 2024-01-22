# %%
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import plotting

from modules import vox2mni, mni2vox

id = "A1BG"

def get_abagen_result(id):
    df = pd.read_csv("./data/MNI152_T1_1mm_brain_abagen_expression_interpolate_10col.csv", index_col=0)
    idx = df.columns.get_loc(id)
    columns = df.columns
    df_tensor = df.iloc[:, idx].values
    index_tensor = df.index.values
    return columns[idx], index_tensor, df_tensor

file = "MNI152_T1_1mm_brain"
# file = "atlas-desikankilliany"
atlas = nib.load(f"./data/{file}.nii.gz")
atlas_data = atlas.get_fdata()
affine = atlas.affine

regions = np.unique(atlas_data)
label_voxel_coordinates = {label: np.argwhere(atlas_data == label) for label in regions}

id, labels, values = get_abagen_result(id)

first = True
for label, value in zip(labels, values):
    if label in label_voxel_coordinates:
        voxel_coords = label_voxel_coordinates[label]
        for coord in voxel_coords:
            atlas_data[tuple(coord)] = value

new_img = nib.Nifti1Image(atlas_data, affine=affine)
view = plotting.view_img(new_img,
                         bg_img=atlas,
                         threshold=0.1,
                         cmap='cold_white_hot_r')

view.save_as_html(f'./{file}_{id}_abagen.html')
nib.save(new_img, f'./{file}_{id}_abagen.nii')


# %%
