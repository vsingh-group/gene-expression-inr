import abagen
import pandas as pd

matter = "246"

def save_results(microarray):
    print(microarray.keys())
    for subj in microarray:
        print(f"{subj}: {microarray[subj].shape}")
        microarray[subj].to_csv(f"./data/abagendata/train/microarray_{subj}.csv")

# nii = f"MNI152_T1_1mm_brain_{matter}_mask_int" #_grey_mask_int
nii = "BN_Atlas_246_1mm"
nii_file = f'./data/atlas/{nii}.nii.gz'

# use all donors, 6 possible
# get nifti from abagen, voxel level
expression = abagen.get_expression_data(nii_file,
                                        # missing='interpolate',
                                        return_donors=True,
                                        region_agg=None,
                                        return_report=False
                                        )
save_results(expression)