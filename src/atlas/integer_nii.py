import nibabel as nib
import numpy as np

nii = "MNI152_T1_1mm_brain_grey_mask" # grey or white
file_path = f'./data/{nii}.nii.gz'
nii_image = nib.load(file_path)

image_data = nii_image.get_fdata()

# Convert the data to integers. You can use np.round() if you want to round instead.
int_image_data = np.rint(image_data).astype(np.int32)

new_nii_image = nib.Nifti1Image(int_image_data, affine=nii_image.affine)

new_file_path = f'./data/{nii}_int.nii.gz'
nib.save(new_nii_image, new_file_path)