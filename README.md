

## Dependencies

`nibabel`, `nilearn`

- `src/atlas/...`
    - `filter_nii.py` - filter atlas nii file under certain threshold, `MNI152_T1_1mm_brain_grey.nii.gz` -> `MNI152_T1_1mm_brain_grey_mask.nii.gz`
    - `integer_nii.py` - convert atlas to integer values to fit abagen input requirement, `MNI152_T1_1mm_brain_grey_mask.nii.gz` -> `MNI152_T1_1mm_brain_grey_mask_int.nii.gz`
- `src/data/...`
    - `generate4d.py` - generate 4-dimentional data for training, that is whether certain point is on white or grey matter, white for 1, grey for -1, neither for 0
    - `pc1_se.py` - generate pc1/spectrum embedding order for relevant genes
    - `data_preprocess.py`
        - calculate mean Microarray result for the same gene from differnent probes
        - merge pc1/spectrum embedding order to gene x y z locations for training
- `src/plots/...`
    - `similarity_gene.py` - get similarity matrix from **only gene values** under `se/pc1` ordering, generate 2 png files
    - `similarity_brain.py` - get similarity matrix from **brain images** under `se/pc1` ordering, generate 2 png files
    - `visualize_abagen.py` - visualize abagen result in nii file
    - `visualize_se.py` - generate git files, compare from separate trained result and whole trained result, under se ordering, require nii files to generated first from `inference.py`
    - `visualize.py` - no interpolation, simply map gene points to nii

- `inference.py` - INR interpolation, require trained pth file
- `main.sh` - training all gene expressions with one command