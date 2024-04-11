
## Env

```
conda env create -f environment.yml
conda activate inr
```

## Dependencies

`nibabel`, `nilearn`

## Documentations

### Data

- `data/abagendata/abagen_output/...` - abagen output data for baseline only, generated from `abagen_compare.py`
    -  `<atlas>_microarray_<donor_id>.csv` - region aggregated abagen output w/o interpolation for <donor_id> on atlas
    -  `<atlas>_interpolation_microarray_<donor_id>.csv` - region aggregated abagen output **with** interpolation for <donor_id> on atlas 
    -  atlas naming:
        - `<atlas> == 246` - `BN_Atlas_246_1mm.nii.gz`
        - `<atlas> == grey` - `MNI152_T1_1mm_brain_grey_mask_int.nii.gz`
        - `<atlas> == white` - `MNI152_T1_1mm_brain_white_mask_int.nii.gz`
- `data/abagendata/train/...` - abagen output data for training without region aggregation, only preprocessed (dropped useless measurements)
    - `microarray`

### Coding

- `src/atlas/...`
    - `filter_nii.py` - filter atlas nii file under certain threshold, `MNI152_T1_1mm_brain_grey.nii.gz` -> `MNI152_T1_1mm_brain_grey_mask.nii.gz`
    - `integer_nii.py` - convert atlas to integer values to fit abagen input requirement, `MNI152_T1_1mm_brain_grey_mask.nii.gz` -> `MNI152_T1_1mm_brain_grey_mask_int.nii.gz`
- `src/data/...`: Get training data, `generate4d.py` -> `pc1_se.py` -> `data_merge.py` -> `data/se_merged.csv`
    - `generate4d.py` - generate 4-dimentional data for training, that is whether certain point is on white or grey matter, white for 1, grey for -1, neither for 0
    - `pc1_se.py` - generate pc1/spectrum embedding order for relevant genes
    - `data_merge.py`
        - merge pc1/spectrum embedding order to gene x y z locations for training
- `src/plots/...`
    - `similarity_gene.py` - get similarity matrix from **only gene values** under `se/pc1` ordering, generate 2 png files
    - `similarity_brain.py` - get similarity matrix from **brain images** under `se/pc1` ordering, generate 2 png files
    - `visualize_abagen.py` - visualize abagen result in nii file
    - `visualize_se.py` - generate git files, compare from separate trained result and whole trained result, under se ordering, require nii files to generated first from `inference.py`
    - `visualize.py` - no interpolation, simply map gene points to nii

- `inference.py` - INR interpolation, require trained pth file
- `main.sh` - training all gene expressions with one command