"""
Create a pool of negative axial slices for the subjects in the training set (1-660).

"""

# Import packages
import numpy as np
import os
import pandas as pd
import nibabel as nib 
import matplotlib.image

# Access data
image_inputpath = "/home/u366836/DATA/LIDC/Masked_Scans" 
label_inputpath = "/home/mleeuwen/DATA/LIDC_data/New_Uniform_labels_nodules_ambiguous" 
all_labels = os.listdir(label_inputpath)
all_labels.sort()

#datasplit = {"train":(1, 660), "val":(661, 827), "test":(828, 1012)}


# Loop over all training set subjects and extract negative slices
print("Creating a pool of negative CT scan slices.", flush=True)
for lab_file in all_labels:
    subject = lab_file[10:14]
    label_nifti = nib.load(label_inputpath+'/'+lab_file)
    label = label_nifti.get_fdata().astype(int)
    lungs = np.load(f"{image_inputpath}/{subject}_lungs.npy")
    
    if int(subject) in range(0, 661): # training set
        for i in range(0,label.shape[2]):
            label_slice = label[:,:,i]
            if label_slice.max() == 0: # there is no nodule in here!
                lung_slice = lungs[:,:,i]
                matplotlib.image.imsave(f'/home/u366836/DATA/LIDC/train_negatives/NEGATIVE_{subject}_top_{i}.png', lung_slice, cmap='gray')
                
    if int(subject) in range(661, 828): # validation set
        for i in range(0,label.shape[2]):
            label_slice = label[:,:,i]
            if label_slice.max() == 0: # there is no nodule in here!
                lung_slice = lungs[:,:,i]
                matplotlib.image.imsave(f'/home/u366836/DATA/LIDC/val_negatives/NEGATIVE_{subject}_top_{i}.png', lung_slice, cmap='gray')
            
    elif int(subject) in range(828, 1013): # test set
        for i in range(0,label.shape[2]):
            label_slice = label[:,:,i]
            if label_slice.max() == 0: # there is no nodule in here!
                lung_slice = lungs[:,:,i]
                matplotlib.image.imsave(f'/home/u366836/DATA/LIDC/test_negatives/NEGATIVE_{subject}_top_{i}.png', lung_slice, cmap='gray')
            
print("Saved all negative slices.", flush=True)  



