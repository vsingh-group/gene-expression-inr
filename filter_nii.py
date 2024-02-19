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

def filter_nii_files(atlas, threshold):
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
                if data[x, y, z] < threshold and data[x, y, z] > -threshold:
                    data[x, y, z] = 0

    new_img = nib.Nifti1Image(data, affine=image.affine)
    nib.save(new_img, f'./data/{atlas}_mask.nii.gz')
    

threshold = 1e-2
filter_nii_files("MNI152_T1_1mm_brain_grey", threshold)
filter_nii_files("MNI152_T1_1mm_brain_white", threshold)