# %%
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt

import pdb

meta_df = pd.read_excel('./data/GeneRegionTable.xlsx')    
gen9m3_df = pd.read_csv('./data/MicroarrayExpression.csv', header=None)
meta_df = meta_df[['mni_x', 'mni_y', 'mni_z']]

gen9m3_df = gen9m3_df.drop(gen9m3_df.columns[0], axis=1) # drop first column

# take first row of gen9m3_df
gen9m3_df.iloc[0, :].values
# add as a new column to meta_df
meta_df['vals'] = gen9m3_df.iloc[0, :].values

nii_file = './data/MNI152_T1_1mm.nii.gz'
image = nib.load(nii_file)
data = image.get_fdata()
affine = image.affine
print(image.header)


# Function to convert MNI coordinates to voxel indices
def mni2vox(mni_coords, affine):
    voxel_coords = np.linalg.inv(affine) @ np.append(mni_coords, 1)
    return np.rint(voxel_coords[:3]).astype(int)

# Create an empty volume
plot_data = np.zeros(data.shape)

# Map the values from the DataFrame to the voxel space
for index, row in meta_df.iterrows():
    voxel_coords = mni2vox([row['mni_x'], row['mni_y'], row['mni_z']], affine)
    if np.all(voxel_coords < np.array(data.shape)) and np.all(voxel_coords >= 0):
        plot_data[tuple(voxel_coords)] = row['vals']

# %%
new_img = nib.Nifti1Image(plot_data, affine=image.affine)
nib.save(new_img, 'MNI152_T1_1mm+gene.nii')

# Plot static 3d image
plotting.plot_stat_map(new_img, bg_img=nii_file, display_mode='ortho', cut_coords=[0, 0, 0], threshold=0.1)
# interactive plot
view = plotting.view_img(new_img, bg_img=nii_file, threshold=0.1)
view.save_as_html(f'./MNI152_T1_1mm+gene.html')
