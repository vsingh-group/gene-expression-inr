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


def get_train_data_sep(order="pc1"):
    if order != "pc1" and order != "se":
        print("Error in choosing gene order")
        exit(1)
    filepath=f'./data/{order}_merged.csv'
    meta_df = pd.read_csv(filepath)

    val_dic = {}
    coords_dic = {}
    
    # Iterate over unique gene symbols
    for gene in meta_df['gene_symbol'].unique():
        # Filter rows for the current gene
        gene_data = meta_df[meta_df['gene_symbol'] == gene]
        
        vals = torch.tensor(gene_data[['value']].values, dtype=torch.float32)
        coords = torch.tensor(gene_data[['mni_x', 'mni_y', 'mni_z', order, 'classification']].values, dtype=torch.float32)
        
        val_dic[gene] = vals
        coords_dic[gene] = coords
        
    return coords_dic, val_dic


class BrainFitting(Dataset):
    def __init__(self, coords, vals, normalize=True):
        super().__init__()
        self.coords, self.vals = coords, vals
        
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

def main(gene_order, gene_symbol, coords, vals):
    try:
        brain = BrainFitting(coords, vals)

        dataloader = DataLoader(brain, batch_size=1, pin_memory=True, num_workers=0)
        brain_siren = Siren(in_features=5, out_features=1, hidden_features=256, 
                            hidden_layers=5, outermost_linear=True)
        brain_siren.cuda()

        total_steps = 1000
        steps_til_summary = 50

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


        os.makedirs('./models_test', exist_ok=True)
        torch.save(brain_siren.state_dict(), f'./models_test/{gene_order}_{gene_symbol}.pth')

        min_max_dict = {
            'gene_symbol': gene_symbol,
            'min_vals': brain.min_vals.numpy().item(),
            'max_vals': brain.max_vals.numpy().item(),
            'min_coords': brain.min_coords.numpy().item(),
            'max_coords': brain.max_coords.numpy().item()
        }
            
        with open(f'./models_test/max_min_values_{gene_order}_sep.csv', 'a') as file:
            writer = csv.writer(file)
            row = [str(value) for value in min_max_dict.values()]
            writer.writerow(row)
            
        logging.info(f"[Success]--{gene_order}--{gene_symbol}")

    except Exception as e:
        logging.error(f"[Error]--{gene_order}--{gene_symbol}--{e}")

if __name__ == "__main__":
    gene_order = "se"    
    coords_dic, val_dic = get_train_data_sep("se")
    
    for key in coords_dic.keys():
        coords = coords_dic[key]
        vals = val_dic[key]
        main(gene_order, key, coords, vals)