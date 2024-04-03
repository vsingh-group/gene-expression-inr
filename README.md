

## Dependencies

`nibabel`, `nilearn`

- `filter_nii.py` - filter atlas nii file under certain threshold
- `generate4d.py` - generate 4-dimentional data for training, that is whether certain point is on white or grey matter, white for 1, grey for -1, neither for 0
- `pc1_se.py` - generate pc1/spectrum embedding order for relevant genes
- `data_preprocess.py`
    - calculate mean Microarray result for the same gene from differnent probes
    - merge pc1/spectrum embedding order to gene x y z locations for training
- `inference.py` - INR interpolation, require trained pth file
- `main.sh` - training all gene expressions with one command