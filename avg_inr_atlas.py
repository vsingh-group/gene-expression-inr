import nibabel as nib
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

def avg_atlas(gene_atlas_path, atlas, donor, matter):
    input_path = f'./nii_{donor}_{matter}/{gene_atlas_path}.nii.gz'
    output_path = f'./nii_{donor}_{matter}/{gene_atlas_path}_avg.nii.gz'
    # if os.path.exists(output_path):
    #     return
    # if not os.path.exists(input_path):
    #     return
    atlas_with_labels = nib.load(f'./data/atlas/{atlas}.nii.gz')
    atlas_with_genes = nib.load(input_path)

    # Get data from images
    label_data = atlas_with_labels.get_fdata()
    gene_data = atlas_with_genes.get_fdata()

    regions = np.unique(label_data)
    label_voxel_coordinates = {label: np.argwhere(label_data == label) for label in regions}
    
    new_gene_data = np.zeros_like(gene_data)
    avg_map = {}
        
    for label, coords in label_voxel_coordinates.items():
        # get average value of coords in atlas_with_gene
        if label == 0:
            continue
        
        gene_values = [gene_data[tuple(coord)] for coord in coords]    
        avg_gene_expression = np.mean(gene_values)
        
        new_gene_data[tuple(zip(*coords))] = avg_gene_expression
        avg_map[int(label)] = avg_gene_expression
    
    new_img = nib.Nifti1Image(new_gene_data, atlas_with_genes.affine)
    nib.save(new_img, output_path)
    # print(f"Interpolated {output_path} Success!")
    return avg_map

def average_2_nii_files(gene, output_file):
    try:
        nii1 = nib.load(os.path.join('result_ibf_2full+mirror_TT/nii_9861_83_new', gene))
        nii2 = nib.load(os.path.join('result_ibf_2full+mirror_TT/nii_10021_83_new', gene))
    except:
        print(f"Error: {gene}")

    data1 = nii1.get_fdata()
    data2 = nii2.get_fdata()

    # Check if the shapes of the two data arrays are the same
    if data1.shape != data2.shape:
        raise ValueError("The input NIfTI files must have the same shape.")

    # Compute the average of the two data arrays
    avg_data = (data1 + data2) / 2

    # Create a new NIfTI image
    avg_img = nib.Nifti1Image(avg_data, affine=nii1.affine, header=nii1.header)

    # Save the new NIfTI image
    nib.save(avg_img, output_file)


with open("./data/gene_names.csv") as f:
    gene_names = f.readlines()
    gene_names = [x.strip() for x in gene_names]
    
gene_names = list(set(gene_names))
gene_names.sort()

matter = "83_new" # "246"
donor = "9861"
# atlas = "BN_Atlas_246_1mm"
atlas = "atlas-desikankilliany"
full_records = True
df = pd.read_csv(f"./data/abagendata/train_{matter}/se_{donor}.csv")
os.makedirs(f"./nii_{donor}_{matter}", exist_ok=True)
genes_data = {}

# for i, row in tqdm(df.iterrows(), total=df.shape[0]):
#     id = row['gene_symbol']
#     path = f"{id}_{matter}_inr"
    
#     if full_records:
#         path = f"{id}_{matter}_inrs"
        
#     avg_map = avg_atlas(path, atlas, donor, matter)
#     genes_data[id] = avg_map

for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    id = row['gene_symbol']
    path = f"{id}_{matter}_inrs.nii.gz"
    average_2_nii_files(path, f"result_ibf_2full+mirror_TT/nii_inrs/{path}.nii.gz")


# df = pd.DataFrame(genes_data)
# df.index.name = 'label'
# df = df.sort_index(axis=1)
# if full_records:
#     df.to_csv(f'./data/result_{matter}_{donor}_inrs_avg.csv')
# else:
#     df.to_csv(f'./data/result_{matter}_{donor}_inr_avg.csv')




# df_abagen = pd.read_csv(f"./data/{matter}_interpolation_abagen_full.csv")
# gene_names.remove('NEAT')
# df_abagen = df_abagen[gene_names]
# df_abagen['label'] = range(1, len(df_abagen) + 1)
# df_abagen.index = df_abagen['label']
# df_abagen.to_csv(f'./data/{matter}_interpolation_abagen.csv')