# %%
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import nibabel as nib

# id = "A1BG"

def get_abagen_result(id, donor, matter):
    df = pd.read_csv(f"./data/abagendata/abagen_output/{matter}_interpolate_microarray_{donor}.csv", index_col=0)
    idx = df.columns.get_loc(id)
    columns = df.columns
    df_tensor = df.iloc[:, idx].values
    index_tensor = df.index.values
    return columns[idx], index_tensor, df_tensor

def map_abagen_to_nii(id, atlas, matter, donor):
    # file = "atlas-desikankilliany"
    atlas = nib.load(f"./data/atlas/{atlas}.nii.gz")
    atlas_data = atlas.get_fdata()
    affine = atlas.affine

    regions = np.unique(atlas_data)
    label_voxel_coordinates = {label: np.argwhere(atlas_data == label) for label in regions}

    id, labels, values = get_abagen_result(id, donor, matter)

    first = True
    for label, value in zip(labels, values):
        if label in label_voxel_coordinates:
            voxel_coords = label_voxel_coordinates[label]
            for coord in voxel_coords:
                atlas_data[tuple(coord)] = value

    new_img = nib.Nifti1Image(atlas_data, affine=affine)
    nib.save(new_img, f'./nii_{donor}_{matter}/{id}_{matter}_abagen.nii.gz')

matter = "grey" # "grey"
donor = "9861"
# atlas = f"MNI152_T1_1mm_brain_{matter}_mask_int"
# atlas = "BN_Atlas_246_1mm"
atlas = 'atlas-desikankilliany'

matter = "83" # 246
df = pd.read_csv(f"./data/abagendata/train_{matter}/se_{donor}.csv")

os.makedirs(f'./nii_{donor}_{matter}', exist_ok=True)

for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    id = row['gene_symbol']
    order_val = row['se']
    map_abagen_to_nii(id, atlas, matter, donor)
# %%
