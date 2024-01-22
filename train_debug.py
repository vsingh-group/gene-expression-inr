import os
import pdb
import csv
import pickle
import logging
import configargparse
import nibabel as nib

import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

from modules import *

def get_abagen_result(idx):
    df = pd.read_csv("./data/MNI152_T1_1mm_brain_abagen_expression_interpolate_10col.csv", index_col=0)
    columns = df.columns
    df_tensor = torch.tensor(df.iloc[:, idx].values, dtype=torch.float32)
    index_tensor = torch.tensor(df.index.values, dtype=torch.float32)
    return columns[idx], index_tensor, df_tensor

def get_brain_coords():
    file = "MNI152_T1_1mm_brain"
    atlas = nib.load(f"./data/{file}.nii.gz")
    atlas_data = atlas.get_fdata()
    affine = atlas.affine
    
    regions = np.unique(atlas_data)
    region_centroids = {}
    
    for region in regions:
        if region == 0:
            continue
        
        voxels = np.argwhere(atlas_data == region)
        centroid = np.mean(voxels, axis=0)
        region_centroids[region] = vox2mni(centroid, affine)
        
    return region_centroids

class BrainFitting(Dataset):
    def __init__(self, idx=0, normalize=True):
        super().__init__()
        self.id, self.region_indices, self.vals = get_abagen_result(idx)
        self.coords = get_brain_coords()
        coords_np = np.array(list(self.coords.values()))
        self.coords = torch.tensor(coords_np, dtype=torch.float32)
        self.vals = self.vals.unsqueeze(1)
        
        if normalize:
            self.vals, self.min_vals, self.max_vals = \
                self.min_max_normalize(self.vals, 0, 1)
            self.coords, self.min_coords, self.max_coords = \
                self.min_max_normalize(self.coords, -1, 1)

    def min_max_normalize(self, tensor, min_range, max_range):
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        normalized_tensor = (tensor - min_val) * (max_range - min_range) \
            / (max_val - min_val) + min_range
        return normalized_tensor, min_val, max_val
    
    def min_max_unnormalize(
        self, normalized_tensor, min_val, max_val, min_range, max_range):
        
        unnormalized_tensor = \
            (normalized_tensor - min_range) * (max_val - min_val) \
            / (max_range - min_range) + min_val
            
        return unnormalized_tensor

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.vals

# logging.basicConfig(filename='./brain_fitting.log', level=logging.INFO, 
#                     format='%(asctime)s %(levelname)s:%(message)s')
# p = configargparse.ArgumentParser()
# p.add_argument('--index', type=int, default=0)
# opt = p.parse_args()

def main():
    try:
        # torch.cuda.set_device(1)
        # pdb.set_trace()
        # brain = BrainFitting(idx=opt.index)
        brain = BrainFitting(idx=9)

        dataloader = DataLoader(brain, batch_size=1, pin_memory=True, num_workers=0)
        brain_siren = Siren(in_features=3, out_features=1, hidden_features=256, 
                            hidden_layers=3, outermost_linear=True)
        brain_siren.cuda()

        total_steps = 500
        steps_til_summary = 10

        optim = torch.optim.Adam(lr=1e-4, params=brain_siren.parameters())

        model_input, ground_truth = next(iter(dataloader))
        model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

        for step in range(total_steps):
            model_output, coords = brain_siren(model_input)
            loss = torch.mean((model_output - ground_truth)**2)
                
            if not step % steps_til_summary:
                print("Step %d, Total loss %0.6f" % (step, loss))

            optim.zero_grad()
            # accelerator.backward(loss)
            loss.backward()
            optim.step()


        torch.save(brain_siren.state_dict(), f'./{brain.id}.pth')

        min_max_dict = {
            'id': brain.id,
            'min_vals': brain.min_vals.numpy().item(),
            'max_vals': brain.max_vals.numpy().item(),
            'min_coords': brain.min_coords.numpy().item(),
            'max_coords': brain.max_coords.numpy().item()
        }
            
        with open(f'./abagen_max_min_values.csv', 'a') as file:
            writer = csv.writer(file)
            row = [str(value) for value in min_max_dict.values()]
            writer.writerow(row)
            
        # logging.info(f"[Success]--{opt.index}--{brain.id}")

    except Exception as e:
        print(f"[Error]--{brain.id}--{e}")

if __name__ == "__main__":
    main()