import torch
import pickle
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from modules import Siren, vox2mni, mni2vox

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# min_max_dict_df_path = "./models_test/max_min_values.csv"
min_max_dict_df_path = "./models_test/max_min_values_se_sep.csv"

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    # 256 for normal gene model, 512 for large net model
    model = Siren(in_features=5, out_features=1, hidden_features=512, hidden_layers=5, outermost_linear=True)
    model.load_state_dict(checkpoint)
    model.eval() 
    return model

def get_result(id, xyz, model, all_records):
    min_max_dict_df = pd.read_csv(min_max_dict_df_path)
    if not all_records:
        min_max_dict_df = min_max_dict_df[min_max_dict_df['id'] == str(id)]   
    else:
        min_max_dict_df = min_max_dict_df[min_max_dict_df['id'] == 'ALL_RECORDS'] 
    min_max_dict = min_max_dict_df.to_dict(orient='records')[0]

    min_vals = torch.tensor(min_max_dict['min_vals'])
    max_vals = torch.tensor(min_max_dict['max_vals'])
    min_coords = torch.tensor(min_max_dict['min_coords'])
    max_coords = torch.tensor(min_max_dict['max_coords'])
    
    def unnormalize_val(tensor):
        return (tensor - 0) * (max_vals - min_vals) / (1 - 0) + min_vals

    def normalize_coord(coords):
        coords_to_normalize = coords[:, :4]
        coords_fixed = coords[:, 4:]
        
        normalized_coords = (coords_to_normalize - min_coords) * (1 - (-1)) / (max_coords - min_coords) - 1
        return torch.cat((normalized_coords, coords_fixed), dim=1)
    
    coords = torch.tensor(xyz, dtype=torch.float32).to(device)
    coords = normalize_coord(coords)
    
    output = model(coords)
    output = unnormalize_val(output[0])
    
    return output[:,0].cpu().detach().numpy()

def get_results(id, xyz, model, all_records=False, batch_size=4096):
    results = []
    for i in range(0, len(xyz), batch_size):
        batch_xyz = xyz[i:i+batch_size]
        batch_results = get_result(id, batch_xyz, model, all_records)
        results.extend(batch_results)
    return np.array(results)

def inference(id, matter, atlas, model_path, all_records=False, order_val=None):
    brain_inr = load_model(model_path).to(device)

    nii_file = f'./data/{atlas}.nii.gz'
    image = nib.load(nii_file)
    data = image.get_fdata()
    affine = image.affine
    # print(image.header)

    x_dim, y_dim, z_dim = 182, 218, 182  # dimensions from the nii file header

    xyz = []
    mni_coords = []
    for x in range(x_dim):
        for y in range(y_dim):
            for z in range(z_dim):
                if data[x, y, z] > 0:
                    xyz.append([x, y, z])
                    mni_coords.append(vox2mni([x, y, z], affine))

    if matter == "white":
        classification_val = 1  # 1 for white
    elif matter == "grey":
        classification_val = -1  # -1 for grey
    else:
        print("Error in Brain Matter selection")
        exit(0)

    mni_coords = [np.append(coord, classification_val) for coord in mni_coords]

    # add oder val
    if order_val:
        mni_coords = [np.append(coord, order_val) for coord in mni_coords]

    print(f"Generating Results for gene {id}...")      
    # chooose xyz list
    outputs = get_results(id, mni_coords, brain_inr, all_records)

    plot_data = np.zeros(data.shape)

    # Map the values from the DataFrame to the voxel space
    for index, coord in enumerate(xyz):
        # if np.all(coord < np.array(data.shape)) and np.all(coord >= 0):
        plot_data[tuple(coord)] = outputs[index]
        
    new_img = nib.Nifti1Image(plot_data, affine=image.affine)
    
    if all_records:
        save_path = f'./nii_results_full/{matter}_{id}.nii.gz'
    else:
        save_path = f'./nii_results_sep/{matter}_{id}.nii.gz'
    nib.save(new_img, save_path)
    print("Interpolate Success!")


# id = "1058685"
matter = "white" # "grey"
atlas = f"MNI152_T1_1mm_brain_{matter}_mask"
all_records = True
df = pd.read_csv("./data/se.csv")

for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    id = row['gene_symbol']
    order_val = row['se']
    if all_records:
        model_path = f'./models_test/model_se_0.0002_512_5.pth'
    else:
        model_path = f'./models_test/se_{id}.pth'
    inference(id, matter, atlas, model_path, all_records, order_val)
