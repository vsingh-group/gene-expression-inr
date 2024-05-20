import os
import pdb
import csv
import logging

import pandas as pd

import wandb
import torch
from torch.utils.data import Dataset, DataLoader

from modules import *


def get_train_data(donor, matter, order="pc1", encoding_dim=4):
    if order != "pc1" and order != "se":
        print("Error in choosing gene order")
        exit(1)
    filepath=f'./data/abagendata/train_{matter}/{order}_{donor}_merged.csv'
    meta_df = pd.read_csv(filepath)
    vals = torch.tensor(meta_df[['value']].values, dtype=torch.float32)
    meta_df = meta_df.drop(['gene_symbol', 'well_id', 'value'], axis=1)
    meta_df = encode_df(meta_df, multires=encoding_dim)
    print(meta_df.head())
    coords = torch.tensor(meta_df.values, dtype=torch.float32)
    return coords, vals


class BrainFitting(Dataset):
    def __init__(self, donor, matter, gene_order, encoding_dim=4, normalize=True):
        super().__init__()
        self.coords, self.vals = get_train_data(donor, matter, gene_order, encoding_dim) # se or pc1
        
        # Assuming the first 3 columns are mni_x, mni_y, mni_z
        self.coords_to_normalize = self.coords[:, :3]  
        self.coords_fixed = self.coords[:, 3:]  # Assuming the last one column is 'order'
        
        if normalize:
            # self.vals, self.min_vals, self.max_vals = \
            #     self.min_max_normalize(self.vals, 0, 1)
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

def train(config, donor):
    try:
        gene_order = config.gene_order
        brain = BrainFitting(donor, config.matter, gene_order, config.encoding_dim)

        dataloader = DataLoader(brain, batch_size=1, pin_memory=True, num_workers=0)
        brain_siren = Siren(in_features=5+config.encoding_dim*2,
                            out_features=1,
                            hidden_features=config.hidden_features, 
                            hidden_layers=config.hidden_layers,
                            outermost_linear=True)
        brain_siren.cuda()

        total_steps = config.total_steps
        steps_til_summary = 50

        optim = torch.optim.Adam(lr=config.lr, params=brain_siren.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=400, gamma=0.9)
        
        model_input, ground_truth = next(iter(dataloader))
        model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

        prev_loss = 1
        os.makedirs('./models_test', exist_ok=True)

        for step in range(total_steps):
            model_output, coords = brain_siren(model_input)
            loss = torch.mean((model_output - ground_truth)**2)
            
            if loss < prev_loss:
                model_path_prefix = (
                    f"./models_test/model_{config.matter}_{config.lr}_"
                    f"{5 + 2 * config.encoding_dim}x"
                    f"{config.hidden_features}x"
                    f"{config.hidden_layers}_"
                )
                
                old_model = f"{model_path_prefix}{prev_loss}.pth"   
                # Remove the old model file
                if os.path.exists(old_model):
                    os.remove(old_model)
                
                # Save the new model file
                torch.save(
                    brain_siren.state_dict(),
                    f"{model_path_prefix}{loss}.pth"
                )
                
                # Update the previous loss
                prev_loss = loss
    
            wandb.log({"loss": loss.item()})
            if not step % steps_til_summary:
                print("Step %d, Total loss %0.6f" % (step, loss))

            optim.zero_grad()
            # accelerator.backward(loss)
            loss.backward()
            optim.step()
            scheduler.step()

        
        min_max_dict = {
            'id': 'ALL_RECORDS',
            'min_vals': 0.0,
            'max_vals': 1.0,
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
    train(wandb.config, donor="9861", gene_order=gene_order)
    
def main():
    # matter: 83 or 246 depends on different atlas
    wandb.init(project="brain_fitting0417", entity="yuxizheng", config={
        "matter": "83",
        "gene_order": "se",
        "lr": 1e-4,
        "hidden_layers": 12,
        "hidden_features": 512,
        "total_steps": 5000,
        "encoding_dim": 8,
    })
    
    train(wandb.config, donor="9861")

if __name__ == "__main__":
    # sweep_configuration = {
    #     "method": "random", # bayes
    #     "metric": {"goal": "minimize", "name": "loss"},
    #     "parameters": {
    #         "lr": {"values": [1e-4]},
    #         "hidden_layers": {"values": [10]},
    #         "hidden_features": {"values": [512]},
    #         "total_steps": 5000,
    #         "encoding_dim": {"values": [5, 6, 7, 8, 9]},
    #     },
    # }

    # sweep_id = wandb.sweep(sweep=sweep_configuration, project="brain-gene-sweep")
    # wandb.agent(sweep_id, function=main_sweep, count=40)
    main()
