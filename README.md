# Gene Expression INR

This repository contains the algorithms implemented in the paper: Brain-wide interpolation and conditioning of gene expression in the human brain using Implicit Neural Representations.

[paper](https://arxiv.org/abs/2506.11158)

## Abstract

In this paper, we study the efficacy and utility of recent advances in non-local, non-linear image interpolation and extrapolation algorithms, specifically, ideas based on Implicit Neural Representations (INR), as a tool for analysis of spatial transcriptomics data. We seek to utilize the microarray gene expression data sparsely sampled in the healthy human brain, and produce fully resolved spatial maps of any given gene across the whole brain at a voxel-level resolution. To do so, we first obtained the 100 top AD risk genes, whose baseline spatial transcriptional profiles were obtained from the Allen Human Brain Atlas (AHBA). We adapted Implicit Neural Representation models so that the pipeline can produce robust voxel-resolution quantitative maps of all genes. We present a variety of experiments using interpolations obtained from Abagen as a baseline/reference.

## Installation

```
conda env create -f environment.yml
conda activate inr
```

## Code

### Data

*Following data are generated from `abagen_compare.py`, with customized abagen codebase*

- `data/abagendata/abagen_output/...*.csv` - abagen output data for baseline only
    -  `<atlas>_microarray_<donor_id>.csv` - region aggregated abagen output w/o interpolation for <donor_id> on atlas
    -  `<atlas>_interpolation_microarray_<donor_id>.csv` - region aggregated abagen output **with** interpolation for <donor_id> on atlas 
    -  atlas naming:
        - `<atlas> == 246` - `BN_Atlas_246_1mm.nii.gz`
        - `<atlas> == grey` - `MNI152_T1_1mm_brain_grey_mask_int.nii.gz`
        - `<atlas> == white` - `MNI152_T1_1mm_brain_white_mask_int.nii.gz`
- `data/abagendata/train/...*.csv` - abagen output data for training without region aggregation, only preprocessed (dropped useless measurements)
    - `microarray_<donor_id>.csv` - abagen preprocessed microarry
    - `annotation_<donor_id>.csv` - abagen preprocessed annotation
    - `annotation_<donor_id>_4d.csv` - abagen preprocessed annotation, adding 4th dimension of classification (whether on grey or white matter), generated from `python src/data/generate4d.py`
    - `pc/se_<donor_id>.csv` - selected disease relevant gene expression names and its data, order by pc1 or spectrum embedding, generated from `python src/data/pc1_se.py`
    - `pc/se_<donor_id>_merged.csv`, merge annotation and selected microarry data, and reformat the structure for model training generated from `python src/data/data_merge.py`

### Results

- `nii_<donor_id>/<gene_symbol>_<atlas>_abagen.nii.gz` - brain atlas, mapped gene expression on corresponding gene symbol name and atlas, generated from `python src/plots/visualize_abagen.py` using `<atlas>_interpolation_microarray_<donor_id>.csv`
- `nii_<donor_id>/<gene_symbol>_<atlas>_inr.nii.gz` - brain atlas, mapped gene expression on corresponding gene symbol name and atlas, generated from `python inference.py` using `model_test/<mode>_<gene_symbol>.pth`, this result interpolates all mni measurements in the brain atlas
- `nii_<donor_id>/<gene_symbol>_<atlas>_inr_avg.nii.gz` - brain atlas, postprocessed with `python src/plots/avg_inr_atlas.py`, which averages regions so we can compare with abagen baseline

### Preprocess, Training, and Inference

- `src/atlas/...`
    - `filter_nii.py` - filter atlas nii file under certain threshold, `MNI152_T1_1mm_brain_grey.nii.gz` -> `MNI152_T1_1mm_brain_grey_mask.nii.gz`
    - `integer_nii.py` - convert atlas to integer values to fit abagen input requirement, `MNI152_T1_1mm_brain_grey_mask.nii.gz` -> `MNI152_T1_1mm_brain_grey_mask_int.nii.gz`
- `src/data/...`: Get training data, pipeline: `generate4d.py` -> `pc1_se.py` -> `data_merge.py` -> `data/abagendata/train/se_<donor_id>_merged.csv`
    - `i_generate4d.py` - generate 4-dimentional data for training, that is whether certain point is on white or grey matter, white for 1, grey for -1, neither for 0
    - `ii_pc1_se.py` - generate pc1/spectrum embedding order for relevant genes
    - `iii_data_merge.py`
        - merge pc1/spectrum embedding order to gene x y z locations for training
    - `iv_encoding.py`
- `src/plots/...`
    - `similarity_gene.py` - get similarity matrix from **only gene values** under `se/pc1` ordering, generate 2 png files
    - `similarity_brain.py` - get similarity matrix from **brain images** under `se/pc1` ordering, generate 2 png files
    - `visualize_abagen.py` - visualize abagen result in nii file
    - `visualize_se.py` - generate git files, compare from separate trained result and whole trained result, under se ordering, require nii files to generated first from `inference.py`
    - `visualize.py` - no interpolation, simply map gene points to nii

- `plot_...` - plotting code for manuscript graph
- `inference.py` - INR interpolation, require trained pth file
- `train_gene_net.py` - Our main result, train all genes in one model with model backbone in your selection, in the paper we mainly reported results from `siren`
- `train_gene_net_sep.py` - result baseline, train one gene on one model with model backbone in your selection
- `train_gene_net_noise.py` - Our main result but adding noise for robustness ablation study, train all genes in one model with model backbone in your selection, in the paper we mainly reported results from `siren`
- `train_abagen.py` - do not use, old code.
- `main.sh` - training all gene expressions with one command

### Data Release

Google Drive: https://drive.google.com/drive/folders/1zi8rKqYVd7GsrcfZ-EuZtzsQwL-zA_3V?usp=drive_link

## Acknowledgements

We make use of code from [Implicit Neural Representations with Periodic Activation Functions](https://github.com/vsitzmann/siren), [WIRE: Wavelet Implicit Neural Representations](https://github.com/vishwa91/wire), and [abagen: A toolbox for the Allen Brain Atlas genetics data](https://github.com/rmarkello/abagen). We gratefully acknowledge their significant contributions to the field and their commitment to making high-quality research code publicly available.

## Reference
If you find our paper helpful and/or use this code, please cite our publication.

```
@misc{yu2025brainwideinterpolationconditioninggene,
      title={Brain-wide interpolation and conditioning of gene expression in the human brain using Implicit Neural Representations}, 
      author={Xizheng Yu and Justin Torok and Sneha Pandya and Sourav Pal and Vikas Singh and Ashish Raj},
      year={2025},
      eprint={2506.11158},
      archivePrefix={arXiv},
      primaryClass={q-bio.GN},
      url={https://arxiv.org/abs/2506.11158}, 
}
```