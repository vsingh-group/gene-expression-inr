import abagen
import pandas as pd

matter = "83"

def save_results(microarray):
    print(microarray.keys())
    for subj in microarray:
        if subj == '9861' or subj == "10021":
            print(f"Skipping {subj} for bidirectional")
            continue
        print(f"{subj}: {microarray[subj].shape}")
        microarray[subj].to_csv(f"./data/abagendata/abagen_output/{matter}_microarray_{subj}.csv")

# nii = f"MNI152_T1_1mm_brain_{matter}_mask_int" #_grey_mask_int
# nii = "BN_Atlas_246_1mm"
nii = "atlas-desikankilliany"
nii_file = f'./data/atlas/{nii}.nii.gz'

# use all donors, 6 possible
# get nifti from abagen, voxel level
expression = abagen.get_expression_data(nii_file,
                                        missing='interpolate',
                                        # lr_mirror='bidirectional',
                                        # return_donors=True,
                                        # region_agg=None,
                                        return_report=False
                                        )
expression.to_csv(f"./data/abagendata/abagen_output/{matter}_interpolation.csv")
save_results(expression)