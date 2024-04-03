# %%
import nibabel as nib
import pickle
from modules import mni2vox 

# Load the NIfTI files for brain, white matter, and grey matter
brain = nib.load("./data/MNI152_T1_1mm_brain.nii.gz")
white_matter = nib.load('./data/MNI152_T1_1mm_brain_white_mask.nii.gz')
grey_matter = nib.load('./data/MNI152_T1_1mm_brain_grey_mask.nii.gz')

# Get the data arrays for each component
brain_data = brain.get_fdata()
white_data = white_matter.get_fdata()
grey_data = grey_matter.get_fdata()

# Load MNI coordinates from a pickle file
with open('./centroid_coords.pkl', 'rb') as file:
    coords = pickle.load(file)
    
voxel_coords = {}
for key, mni_coord in coords.items():
    # Convert MNI coordinates to voxel coordinates
    voxel_coords[key] = mni2vox(mni_coord, brain.affine)  # Fixed: brain.affine is the correct usage

# Function to check if a point is in white matter, grey matter, or neither
def check_matter_type(voxel_coord, white_data, grey_data):
    x, y, z = voxel_coord
    if white_data[int(x), int(y), int(z)] > 0:
        return 1  # Indicates white matter
    elif grey_data[int(x), int(y), int(z)] > 0:
        return -1  # Indicates grey matter
    else:
        return 0  # Indicates neither

# Dictionary to hold the classification for each coordinate
matter_classification = {}
for key, voxel_coord in voxel_coords.items():
    matter_classification[key] = check_matter_type(voxel_coord, white_data, grey_data)

# %%
import pandas as pd

# meta_df = pd.read_excel("./data/GeneRegionTable.xlsx")
meta_df = pd.read_csv("data/donor9861/SampleAnnot.csv")
coords = meta_df[['mni_x', 'mni_y', 'mni_z']].values

classifications = []
for mni_coord in coords:
    voxel_coord = mni2vox(mni_coord, brain.affine)
    classification = check_matter_type(voxel_coord, white_data, grey_data)
    classifications.append(classification)

meta_df['classification'] = classifications
meta_df.to_csv("./data/donor9861/SampleAnnot4d.csv", index=False)

# %%
