

## Dependencies

`nibabel`, `nilearn`

`filter_nii.py` - filter atlas nii file under certain threshold
`generate4d.py` - generate 4-dimentional data for training, that is whether certain point is on white or grey matter, white for 1, grey for -1, neither for 0
`inference.py` - INR interpolation, require trained pth file

`main.sh` - training all gene expressions with one command