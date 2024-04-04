import nibabel as nib
import numpy as np

def integer_nii(nii):
    file_path = f'./data/atlas/{nii}.nii.gz'
    nii_image = nib.load(file_path)

    image_data = nii_image.get_fdata()
    regions = np.unique(image_data)
    
    print(f"{len(regions)} Regions")
    
    label_to_int = {label: i for i, label in enumerate(regions)}

    int_image_data = np.zeros_like(image_data, dtype=np.int32)

    # for label in regions:
    #     voxels = np.argwhere(image_data == label)
        
    #     for voxel in voxels:
    #         int_image_data[tuple(voxel)] = int(np.rint(label))        

    for label, new_label in label_to_int.items():
        int_image_data[image_data == label] = new_label
        
    new_nii_image = nib.Nifti1Image(int_image_data, affine=nii_image.affine)

    new_file_path = f'./data/{nii}_mask_int.nii.gz'
    nib.save(new_nii_image, new_file_path)

# integer_nii("MNI152_T1_1mm_brain")
integer_nii("MNI152_T1_1mm_brain_white")
integer_nii("MNI152_T1_1mm_brain_grey")