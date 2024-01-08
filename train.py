import os
import pdb
import csv
import pickle
import logging
import configargparse

import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

from modules import *

def get_brain_coords(filepath='./data/GeneRegionTable.xlsx'):
    """Get brain region spatial coordinates.

    Parameters
    ----------
    filepath : str, optional

    Returns
    -------
    coords : torch.tensor, shape=(n_regions, 3)
        spatial coordinates in tensor form for pytorch model
    """
    # ['structure_id', 'Regions', 'mni_x', 'mni_y', 'mni_z',
    #  'structure_acronym', 'structure_name']
    meta_df = pd.read_excel(filepath)
    
    coords = torch.tensor(meta_df[['mni_x', 'mni_y', 'mni_z']].values, dtype=torch.float32)
    
    return coords

def get_gen9m3(idx=0, filepath='./data/MicroarrayExpression.csv'):
    """Get microarray expressions.

    Parameters
    ----------
    filepath : str, optional

    Returns
    -------
    ids : str
        _description_
    gen9m3 : torch.tensor
        _description_
    """
    # ['gene_id', 'spatial_1', 'spatial_2', ...]
    gen9m3_df = pd.read_csv(filepath, header=None)
    
    # convert to tensors
    ids = gen9m3_df.iloc[:, 0].values.tolist()
    gen9m3 = torch.tensor(gen9m3_df.iloc[:, 1:].values, dtype=torch.float32)
    gen9m3 = gen9m3.unsqueeze(dim=2)
    return ids[idx], gen9m3[idx, :]


class BrainFitting(Dataset):
    def __init__(self, idx=0, normalize=True):
        super().__init__()
        self.id, self.vals = get_gen9m3(idx)
        self.coords = get_brain_coords()
        
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

logging.basicConfig(filename='./brain_fitting.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s')

p = configargparse.ArgumentParser()
p.add_argument('--index', type=int, default=0)
opt = p.parse_args()

def main():
    try:
        # torch.cuda.set_device(1)
        # pdb.set_trace()
        brain = BrainFitting(idx=opt.index)
        # brain = BrainFitting(idx=0)

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


        os.makedirs('./models', exist_ok=True)
        torch.save(brain_siren.state_dict(), f'./models/{brain.id}.pth')

        min_max_dict = {
            'index': opt.index,
            'id': brain.id,
            'min_vals': brain.min_vals.numpy().item(),
            'max_vals': brain.max_vals.numpy().item(),
            'min_coords': brain.min_coords.numpy().item(),
            'max_coords': brain.max_coords.numpy().item()
        }
            
        with open(f'./models/max_min_values.csv', 'a') as file:
            writer = csv.writer(file)
            row = [str(value) for value in min_max_dict.values()]
            writer.writerow(row)
            
        logging.info(f"[Success]--{opt.index}--{brain.id}")

    except Exception as e:
        logging.error(f"[Error]--{opt.index}--{brain.id}--{e}")

if __name__ == "__main__":
    main()