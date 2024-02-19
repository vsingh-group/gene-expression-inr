import torch
import pickle
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from modules import Siren, vox2mni, mni2vox


def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model = Siren(in_features=4, out_features=1, hidden_features=256, hidden_layers=3, outermost_linear=True)
    model.load_state_dict(checkpoint)
    model.eval() 
    return model


def get_result(id, xyz, model):
    min_max_dict_df = pd.read_csv("./models_test/max_min_values.csv")
    min_max_dict = min_max_dict_df[min_max_dict_df['id'] == int(id)]    
    min_max_dict = min_max_dict.to_dict(orient='records')[0]

    min_vals = torch.tensor(min_max_dict['min_vals'])
    max_vals = torch.tensor(min_max_dict['max_vals'])
    min_coords = torch.tensor(min_max_dict['min_coords'])
    max_coords = torch.tensor(min_max_dict['max_coords'])
    
    def unnormalize_val(tensor):
        return (tensor - 0) * (max_vals - min_vals) / (1 - 0) + min_vals

    def normalize_coord(tensor):
        return (tensor - min_coords) * (1 - (-1)) / (max_coords - min_coords) - 1
    
    coords = torch.tensor(xyz, dtype=torch.float32).to(device)
    coords = normalize_coord(coords)
    
    output = model(coords)
    output = unnormalize_val(output[0])
    
    return output[:,0].cpu().detach().numpy()

def get_results(id, xyz, model, batch_size=4096):
    results = []
    for i in range(0, len(xyz), batch_size):
        batch_xyz = xyz[i:i+batch_size]
        batch_results = get_result(id, batch_xyz, model)
        results.extend(batch_results)
    return np.array(results)

id = "1058685"
# id = "AADAT"
atlas = "MNI152_T1_1mm_brain_white_mask"
# atlas = "MNI152_T1_1mm"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model_path = f'./models/brain_siren_{id}.pth'
model_path = f'./models_test/{id}.pth'
brain_inr = load_model(model_path).to(device)

nii_file = f'./data/{atlas}.nii.gz'
image = nib.load(nii_file)
data = image.get_fdata()
affine = image.affine
header = image.header
print(header)

x_dim, y_dim, z_dim = 182, 218, 182  # dimensions from the nii file header

xyz = []
mni_coords = []
for x in range(x_dim):
    for y in range(y_dim):
        for z in range(z_dim):
            if data[x, y, z] > 0:
                xyz.append([x, y, z])
                mni_coords.append(vox2mni([x, y, z], affine))
        
mni_coords = [np.append(coord, 1) for coord in mni_coords] # 1 for white

print("Generating Results...")      
# chooose xyz list
outputs = get_results(id, mni_coords, brain_inr)

plot_data = np.zeros(data.shape)

# Map the values from the DataFrame to the voxel space
for index, coord in enumerate(xyz):
    # if np.all(coord < np.array(data.shape)) and np.all(coord >= 0):
    plot_data[tuple(coord)] = outputs[index]
    
        
new_img = nib.Nifti1Image(plot_data, affine=image.affine)
# view = plotting.view_img(new_img,
#                          bg_img=nii_file,
#                          threshold=1e-6,
#                          vmin=0)


# view.save_as_html(f'./{atlas}_{id}_mask.html')
nib.save(new_img, f'./{atlas}_{id}.nii.gz')
print("Interpolate Success!")