# %%
import nibabel as nib
from mayavi import mlab
from nilearn import plotting
import matplotlib.pyplot as plt

import pdb

nii_file = './data/BN_Atlas_246_1mm.nii.gz'
image = nib.load(nii_file)
data = image.get_fdata()

# plotting.view_img(image, bg_img=False, title='3D Brain Plot', display_mode='ortho')
mlab.figure(bgcolor=(0, 0, 0), size=(800, 800))
mlab.contour3d(data, contours=4, transparent=True)
mlab.show()

# %%
