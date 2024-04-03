import os
import pdb
import csv
import pickle
import logging
import configargparse

import pandas as pd
import matplotlib.pyplot as plt

import wandb
import torch
from torch.utils.data import Dataset, DataLoader

from modules import *


def get_train_data(order="pc1"):
    if order != "pc1" and order != "se":
        print("Error in choosing gene order")
        exit(1)
    filepath=f'./data/{order}_merged.csv'
    meta_df = pd.read_csv(filepath)
    vals = torch.tensor(meta_df[['value']].values, dtype=torch.float32)
    coords = torch.tensor(meta_df[['mni_x', 'mni_y', 'mni_z', 'classification', order]].values, dtype=torch.float32)
    return coords, vals


class BrainFitting(Dataset):
    def __init__(self, gene_order, normalize=True):
        super().__init__()
        self.coords, self.vals = get_train_data(gene_order) # se or pc1
        
        # Assuming the first four columns are mni_x, mni_y, mni_z, classification
        self.coords_to_normalize = self.coords[:, :4]  
        self.coords_fixed = self.coords[:, 4:]  # Assuming the last one column is 'order'
        
        if normalize:
            self.vals, self.min_vals, self.max_vals = \
                self.min_max_normalize(self.vals, 0, 1)
            self.coords_to_normalize, self.min_coords, self.max_coords = \
                self.min_max_normalize(self.coords_to_normalize, -1, 1)
                
            self.coords = torch.cat((self.coords_to_normalize, self.coords_fixed), dim=1)

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

def train(config, gene_order):
    try:
        brain = BrainFitting(gene_order)

        dataloader = DataLoader(brain, batch_size=1, pin_memory=True, num_workers=0)
        brain_siren = Siren(in_features=5, out_features=1,
                            hidden_features=config.hidden_features, 
                            hidden_layers=config.hidden_layers,
                            outermost_linear=True)
        brain_siren.cuda()

        total_steps = config.total_steps
        steps_til_summary = 50

        optim = torch.optim.Adam(lr=config.lr, params=brain_siren.parameters())

        model_input, ground_truth = next(iter(dataloader))
        model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

        prev_loss = 1
        os.makedirs('./models_test', exist_ok=True)

        for step in range(total_steps):
            model_output, coords = brain_siren(model_input)
            loss = torch.mean((model_output - ground_truth)**2)
            
            if loss < prev_loss:
                torch.save(
                    brain_siren.state_dict(),
                    f'./models_test/model_{gene_order}_{config.lr}_{config.hidden_features}_{config.hidden_layers}.pth'
                )
            prev_loss = loss
            
            wandb.log({"loss": loss.item()})
            if not step % steps_til_summary:
                print("Step %d, Total loss %0.6f" % (step, loss))

            optim.zero_grad()
            # accelerator.backward(loss)
            loss.backward()
            optim.step()

        
        min_max_dict = {
            'gene_symbol': 'ALL_RECORDS',
            'min_vals': brain.min_vals.numpy().item(),
            'max_vals': brain.max_vals.numpy().item(),
            'min_coords': brain.min_coords.numpy().item(),
            'max_coords': brain.max_coords.numpy().item()
        }
            
        with open(f'./models_test/max_min_values_{gene_order}_sep.csv', 'a') as file:
            writer = csv.writer(file)
            row = [str(value) for value in min_max_dict.values()]
            writer.writerow(row)
            
        logging.info(f"[Success]--{gene_order}")

    except Exception as e:
        logging.error(f"[Error]--{gene_order}--{e}")
 
def main_sweep():
    wandb.init(project="brain_fitting", entity="yuxizheng")
    gene_order = "se"
    train(wandb.config, gene_order=gene_order)
    
def main():
    wandb.init(project="brain_fitting", entity="yuxizheng", config={
        "lr": 2e-4,
        "hidden_layers": 5,
        "hidden_features": 512,
        "total_steps": 5000
    })
    
    gene_order = "se"
    train(wandb.config, gene_order=gene_order)

if __name__ == "__main__":
    # sweep_configuration = {
    #     "method": "random", # bayes
    #     "metric": {"goal": "minimize", "name": "loss"},
    #     "parameters": {
    #         "lr": {"values": [1e-4, 2e-4, 3e-4]},
    #         "hidden_layers": {"values": [5, 7]},
    #         "hidden_features": {"values": [512, 1024]},
    #         "total_steps": 5000
    #     },
    # }

    # sweep_id = wandb.sweep(sweep=sweep_configuration, project="brain-gene-sweep")
    # wandb.agent(sweep_id, function=main_sweep, count=40)
    main()
