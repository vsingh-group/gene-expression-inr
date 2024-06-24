import os
import pdb
import csv
import logging
import traceback


import pandas as pd

import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from modules import models
from modules.my_modules import *

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
    def __init__(self, donor, matter, gene_order, encoding_dim=4, normalize=True, test=False):
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
            
        if test:
            coords_train, coords_test, vals_train, vals_test = train_test_split(self.coords, self.vals, test_size=0.1, random_state=42)
            coords_val, coords_test, vals_val, vals_test = train_test_split(coords_test, vals_test, test_size=0.5, random_state=42)

            self.train_data = (coords_train, vals_train)
            self.val_data = (coords_val, vals_val)
            self.test_data = (coords_test, vals_test)
        else:
            self.train_data = (self.coords, self.vals)

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
    
    def get_val_data(self):
        return self.val_data

    def get_test_data(self):
        return self.test_data

logging.basicConfig(filename='./brain_fitting.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s')

def train(config):
    try:
        test = False
        brain = BrainFitting(config.donor,
                             config.matter,
                             config.gene_order, 
                             config.encoding_dim,
                             test=test)
        if test:
            val_data = brain.get_val_data()
            test_data = brain.get_test_data()
        train_dataloader = DataLoader(brain,
                                      batch_size=1,
                                      pin_memory=True,
                                      num_workers=0)
        

        model = models.get_INR(
                nonlin=config.nonlin,
                in_features=5+config.encoding_dim*2,
                out_features=1,
                hidden_features=config.hidden_features,
                hidden_layers=config.hidden_layers,
                scale=5.0,
                pos_encode=False,
                sidelength=config.hidden_features)
        device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        model.to(device)

        total_steps = config.total_steps
        steps_til_summary = 50

        optim = torch.optim.Adam(lr=config.lr, params=model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=150, gamma=0.9)
        
        model_input, ground_truth = next(iter(train_dataloader))
        model_input, ground_truth = model_input.to(device), ground_truth.to(device)

        prev_loss = 1
        os.makedirs('./models_test', exist_ok=True)

        for step in range(total_steps):
            model_output, coords = model(model_input)
            loss = torch.mean((model_output - ground_truth)**2)
            
            # Validation step
            if test:
                model.eval()
                with torch.no_grad():
                    val_input, val_truth = val_data[0].to(device), val_data[1].to(device)
                    val_output, _ = model(val_input)
                    val_loss = torch.mean((val_output - val_truth) ** 2)
                model.train()
            
            if loss < prev_loss:
                model_path_prefix = (
                    f"./models_new/{config.nonlin}_{config.matter}_{config.donor}_{config.lr}_"
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
                    model.state_dict(),
                    f"{model_path_prefix}{loss}.pth"
                )
                
                # Update the previous loss
                prev_loss = loss

            if test:
                wandb.log({"loss": loss.item(), "val_loss": val_loss.item()})
            else:
                wandb.log({"loss": loss.item()})
                
            if not step % steps_til_summary:
                print("Step %d, Total loss %0.6f" % (step, loss))

            optim.zero_grad()
            # accelerator.backward(loss)
            loss.backward()
            optim.step()
            scheduler.step()

        if test:
            model.eval()
            with torch.no_grad():
                test_input, test_truth = test_data[0].to(device), test_data[1].to(device)
                test_output, _ = model(test_input)
                test_loss = torch.mean((test_output - test_truth) ** 2)
            model.train()

            print(f"Final Test Loss: {test_loss:.6f}")
            wandb.log({"test_loss": test_loss.item()})
        
        min_max_dict = {
            'id': 'ALL_RECORDS',
            'min_vals': 0.0,
            'max_vals': 1.0,
            'min_coords': brain.min_coords.numpy().item(),
            'max_coords': brain.max_coords.numpy().item()
        }
            
        with open(f'./models_new/max_min_values_{config.gene_order}_sep.csv', 'a') as file:
            writer = csv.writer(file)
            row = [str(value) for value in min_max_dict.values()]
            writer.writerow(row)
            
        logging.info(f"[Success]--{config.gene_order}")

    except Exception as e:
        print(e)
        traceback.print_exc()
        logging.error(f"[Error]--{config.gene_order}--{e}")
 
    
def main():
    # matter: 83 or 246 depends on different atlas
    wandb.init(project="brain_111", entity="yuxizheng", config={
        "nonlin": 'siren', # 'wire' 'gauss' 'mfn' 'relu' 'siren' 'wire2d'
        "matter": "83_new",
        "gene_order": "se",
        "lr": 1e-4,
        "hidden_layers": 12,
        "hidden_features": 512,
        "total_steps": 10000,
        "encoding_dim": 11,
        "donor": "10021",
        "device": 'cuda:0',
    })
    
    train(wandb.config)
    
def main_sweep():
    with wandb.init(project="brain_111", entity="yuxizheng") as run:
        config = run.config
        
        run_name = f"{config.nonlin}_{config.donor}_{config.lr}"
        
        run.name = run_name
        run.save()
        
        train(wandb.config)

if __name__ == "__main__":
    # sweep_configuration = {
    #     "method": "grid", # bayes
    #     "metric": {"goal": "minimize", "name": "loss"},
    #     "parameters": {
    #         "nonlin": {"values": ['siren', 'wire', 'gauss', 'relu']},
    #         "matter": {"values": ['83']},
    #         "gene_order": {"values": ['se']},
    #         "lr": {"values": [1e-3, 3e-3, 5e-3, 1e-4,]},
    #         "hidden_layers": {"values": [12]},
    #         "hidden_features": {"values": [512]},
    #         "total_steps": {"values": [10000]},
    #         "encoding_dim": {"values": [11]},
    #         # "donor": {"values": ['9861']},
    #         "donor": {"values": ['9861', '10021', '12876', '14380', '15496', '15697']},
    #         "device": {"values": ['cuda:3']},
    #     },
    # }

    # sweep_id = wandb.sweep(sweep=sweep_configuration, project="brain_sweep")
    # wandb.agent(sweep_id, function=main_sweep)
    main()
