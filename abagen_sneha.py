import abagen
import os

# Follow guidelines in https://abagen.readthedocs.io/en/stable/cli.html website for abagen usage

# ABA Gene Data Information
# Subject	| Number of Sampling Locations | Half or Whole Brain
# H0351.1009	| 363	| Half
# H0351.1012	| 529	| Half
# H0351.1015	| 470	| Half
# H0351.1016	| 501	| Half
# H0351.2001	| 946	| Whole
# H0351.2002	| 893	| Whole

# Gene expression options
output_dir = './outputs/' # Where you plan to save your output mapped expressions.
atlas = './data/atlas/atlas-desikankilliany.nii.gz' # Atlas you are using for mapping gene expression. 
atlas_info = './data/atlas86_info_abagen.csv' # Atlas info about hemisphere and regions as outlined in abagen website. Enter complete path to '{...}'
data_dir = '/Users/xizheng/abagen-data/microarray' # directory containing downloaded microarray expressions. Complete the path in '{...}' 
donor_list = ['9861', '10021']  # donor IDs with both hemispheres: ['H0351.2001', 'H0351.2002']/ ['9861', '10021'] or use 'all' for all donors
lr_mirror_opt = 'bidirectional'  # Options: None, 'bidirectional', 'leftright', 'rightleft'
# Whether to perform gene normalization (gene_norm) within structural classes (i.e., cortex, subcortex/brainstem, cerebellum) instead of across all available samples.
norm_structures_opt = True
reannotated_opt = True
ibf_threshold_opt = 0.5  # 0 to have no background noise filtering
genes_save_csv_filename = 'abagen_sneha_output.csv' # Complete the filename in '{...}'
genes_save_csv = os.path.join(output_dir, genes_save_csv_filename)

# Fetch gene expression
# gene_expression = abagen.get_expression_data(atlas=atlas, atlas_info=atlas_info, data_dir=data_dir, donors=donor_list,
#                                         reannotated=reannotated_opt, lr_mirror=lr_mirror_opt,
#                                         ibf_threshold=ibf_threshold_opt, norm_structures=norm_structures_opt,
#                                         missing='interpolate',
#                                         return_report=True)

gene_expression = abagen.get_expression_data(atlas,
                                             donors=donor_list,
                                             missing='interpolate',
                                             ibf_threshold=ibf_threshold_opt,
                                             norm_structures=norm_structures_opt,
                                             reannotated=reannotated_opt,
                                             lr_mirror='bidirectional',
                                             return_report=False,
                                            #  region_agg=None,
                                             )

# Gene names to be mapped
# gene_names = ['KIF5A', 'TNIP1', 'C9orf72', 'TBK1', 'UNC13A', 'C21orf2', 'CHCHD10', 'TUBA4A', 'CCNF', 'MATR3',
#                        'NEK1', 'ANXA11', 'TIA1', 'SOD1', 'FUS', 'UBQLN2', 'DCTN1', 'ANG', 'TARDBP', 'VCP',
#                        'OPTN', 'SQSTM1', 'PFN1', 'MAPT', 'BNIP1']

# gene_expression = gene_expression[0][gene_names]
# gene_expression = gene_expression[0]

# Save/export pandas dataframe to csv
gene_expression.to_csv(genes_save_csv, encoding='utf-8', index=None, header=True)