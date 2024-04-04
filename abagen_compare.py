import abagen
import pandas as pd

def save_results(microarray):
    print(microarray.keys())
    for subj in microarray:
        print(f"{subj}: {microarray[subj].shape}")
        microarray[subj].to_csv(f"./data/abagendata/avg_interpolate_microarray_{subj}.csv")

nii = "MNI152_T1_1mm_brain_grey_mask_int" #_grey_mask_int
# nii = "atlas-desikankilliany"
nii_file = f'./data/{nii}.nii.gz'

# use all donors, 6 possible
# get nifti from abagen, voxel level
expression = abagen.get_expression_data(nii_file,
                                        missing='interpolate',
                                        return_donors=True,
                                        # region_agg=None,
                                        return_report=False
                                        )
save_results(expression)