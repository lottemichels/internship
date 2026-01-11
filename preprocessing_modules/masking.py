"""
Function to extract the lungs from a lung CT scan by applying a binary mask to the original image. 

"""

# Import packages
import numpy as np
import nibabel as nib # NIFTI file processing

def apply_lungmask(image, mask):
    # img = nib.load(image)
    # img_arr = img.get_fdata()
    img_arr = np.load(image)
    mask = nib.load(mask)
    mask_arr = mask.get_fdata()
    binarized_mask_arr = mask_arr > 0
    lungs = img_arr * binarized_mask_arr
    return lungs 
