import abagen
import pandas as pd

nii = "MNI152_T1_1mm_brain"
nii_file = f'./data/{nii}.nii.gz'

# expression = abagen.get_expression_data(nii_file)
# expression.to_csv(f"{nii}_abagen_expression.csv")

expression_interpolate = abagen.get_expression_data(nii_file,
                                                    donors=['9861', '10021'],
                                                    missing='interpolate',
                                                    donor_probes='aggregate',
                                                    probe_selection='average')
expression_interpolate.to_csv(f"{nii}_abagen.csv")

df = pd.read_csv(f'{nii}_abagen.csv', index_col=0)
df.iloc[:, :10].to_csv(f'{nii}_abagen_10col.csv', index=True)

