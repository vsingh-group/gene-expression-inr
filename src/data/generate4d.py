# %%
import numpy as np
import pandas as pd
import nibabel as nib
import pickle

def mni2vox(mni_coords, affine):
    voxel_coords = np.linalg.inv(affine) @ np.append(mni_coords, 1)
    return np.rint(voxel_coords[:3]).astype(int)

def generate4d(filename):
    # Load the NIfTI files for brain, white matter, and grey matter
    brain = nib.load("./data/atlas/MNI152_T1_1mm_brain.nii.gz")
    white_matter = nib.load('./data/atlas/MNI152_T1_1mm_brain_white_mask.nii.gz')
    grey_matter = nib.load('./data/atlas/MNI152_T1_1mm_brain_grey_mask.nii.gz')

    # Get the data arrays for each component
    brain_data = brain.get_fdata()
    white_data = white_matter.get_fdata()
    grey_data = grey_matter.get_fdata()
        
    # Function to check if a point is in white matter, grey matter, or neither
    def check_matter_type(voxel_coord, white_data, grey_data):
        x, y, z = voxel_coord
        if white_data[int(x), int(y), int(z)] > 0:
            return 1  # Indicates white matter
        elif grey_data[int(x), int(y), int(z)] > 0:
            return -1  # Indicates grey matter
        else:
            return 0  # Indicates neither

    meta_df = pd.read_csv(f"{filename}.csv")
    coords = meta_df[['mni_x', 'mni_y', 'mni_z']].values

    classifications = []
    for mni_coord in coords:
        voxel_coord = mni2vox(mni_coord, brain.affine)
        classification = check_matter_type(voxel_coord, white_data, grey_data)
        classifications.append(classification)

    meta_df['classification'] = classifications
    meta_df.to_csv(f"{filename}_4d.csv", index=False)


generate4d("./data/abagendata/annot_9861")
generate4d("./data/abagendata/annot_10021")
generate4d("./data/abagendata/annot_12876")
generate4d("./data/abagendata/annot_14380")
generate4d("./data/abagendata/annot_15496")
generate4d("./data/abagendata/annot_15697")