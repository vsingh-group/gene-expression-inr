import abagen
import pandas as pd

nii = "MNI152_T1_1mm_brain"
nii_file = f'./data/{nii}.nii.gz'

expression = abagen.get_expression_data(nii_file)
expression.to_csv(f"{nii}_abagen_expression.csv")

expression_interpolate = abagen.get_expression_data(nii_file, missing='interpolate')
expression_interpolate.to_csv(f"{nii}_abagen_expression_interpolate.csv")

