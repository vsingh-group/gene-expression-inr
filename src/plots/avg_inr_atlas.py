import nibabel as nib
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

def avg_atlas(gene_atlas_path, donor):
    input_path = f'./nii_{donor}/{gene_atlas_path}.nii.gz'
    output_path = f'./nii_{donor}/{gene_atlas_path}_avg.nii.gz'
    # if os.path.exists(output_path):
    #     return
    # if not os.path.exists(input_path):
    #     return
    atlas_with_labels = nib.load('./data/atlas/BN_Atlas_246_1mm.nii.gz')
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
        
        # new_gene_data[tuple(zip(*coords))] = avg_gene_expression
        avg_map[int(label)] = avg_gene_expression
    
    # new_img = nib.Nifti1Image(new_gene_data, atlas_with_genes.affine)
    # nib.save(new_img, output_path)
    
    return avg_map

with open("./data/gene_names.csv") as f:
    gene_names = f.readlines()
    gene_names = [x.strip() for x in gene_names]
    
gene_names = list(set(gene_names))
gene_names.sort()

matter = "246"
donor = "9861"
df = pd.read_csv(f"./data/abagendata/train/se_{donor}.csv")
os.makedirs(f"./nii_{donor}", exist_ok=True)
genes_data = {}

for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    id = row['gene_symbol']
    path = f"{id}_{matter}_inr"
    avg_map = avg_atlas(path, donor)
    genes_data[id] = avg_map

df = pd.DataFrame(genes_data)
df.index.name = 'label'
df = df.sort_index(axis=1)
df.to_csv(f'./data/result_246_{donor}_inravg.csv')

df_abagen = pd.read_csv(f"./data/abagendata/abagen_output/246_interpolate_microarray_{donor}.csv", index_col='label')
gene_names.remove('NEAT')
df_abagen = df_abagen[gene_names]
df_abagen.to_csv(f'./data/result_246_{donor}_abagen.csv')