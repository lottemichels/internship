"""
This script runs the following pre-processing pipeline for LIDC CT scans: 

1. Scaling: apply a lung window to the CT scan and normalize the HU values within the window range
2. Lung masking: ROI extraction
3. Slicing: create 2D CT scan slices and extracting their lesion bounding boxes
4. Converting: putting datafolders and bounding boxes in correct YOLO format

Comment out those steps you don't want to execute/re-run.

"""

# Import packages
import os
import numpy as np
import pandas as pd
import nibabel as nib
import skimage as skim 

# Defining datasplit (global variable)
datasplit = {"train":(1, 660), "val":(661, 827), "test":(828, 1012)}



"""--------------------------------------  Step 1: Scaling --------------------------------------------------"""
from preprocessing_modules.scaling import apply_window, apply_normalization

# Access data 
inputpath = "/home/mleeuwen/DATA/LIDC_data/Images"
outputpath = "/home/u366836/DATA/LIDC/Scaled_Scans" 
all_images = os.listdir(inputpath)
all_images.sort()

# Loop over and process all subjects
print("Scaling all CT scans.", flush=True)
for img_file in all_images: 
    subject = img_file[10:14] 
    img = nib.load(inputpath+'/'+img_file)
    img_arr = img.get_fdata()
    windowed_img = apply_window(img_arr, W=1500, L=-600) # to check: print(windowed_img.min(), windowed_img.max(), flush=True) # should be between -1350 and 150
    normalized_img = apply_normalization(windowed_img) # to check: print(normalized_img.min(), normalized_img.max(), flush=True) # should be between 0 and 1
    np.save(f"{outputpath}/{subject}_scaled.npy", normalized_img)
print("Completed scaling of all CT scans.", flush=True)



"""------------------------------------------  Step 2: Masking ----------------------------------------------"""
from preprocessing_modules.masking import apply_lungmask

# Access data 
image_inputpath = "/home/u366836/DATA/LIDC/Scaled_Scans"
mask_inputpath = "/home/mleeuwen/DATA/LIDC_data/Lung_masks_LIDC"
outputpath = "/home/u366836/DATA/LIDC/Masked_Scans"
all_images = os.listdir(image_inputpath)
all_images.sort()
all_masks = os.listdir(mask_inputpath)
all_masks.sort()

# Loop over and process all subjects
print("Applying lung masking to all CT scans.", flush=True)
for img_file, mask_file in zip(all_images, all_masks): 
    subject = img_file[0:4] 
    lungs = apply_lungmask(image_inputpath+'/'+img_file, mask_inputpath+'/'+mask_file)
    np.save(f"{outputpath}/{subject}_lungs.npy", lungs)
print("Completed lung masking to all CT scans.", flush=True)
    
    
    
"""---------------------------------------- Step 3: Slicing --------------------------------------"""
from preprocessing_modules.slicing import create_slices 

# Access data
image_inputpath = "/home/u366836/DATA/LIDC/Masked_Scans" 
label_inputpath = "/home/mleeuwen/DATA/LIDC_data/New_Uniform_labels_nodules_ambiguous"
slice_outputpath = "/home/u366836/DATA/LIDC/YOLO_Data_EXP1/images"
props_outputpath = "/home/u366836/DATA/LIDC/Nodule_properties_per_slice.csv" # this remains constant between experiments
all_labels = os.listdir(label_inputpath)
all_labels.sort()
print(f"Found {len(os.listdir(image_inputpath))} lungfiles and {len(all_labels)} labels.")

# Define desired slice properties
properties = ['label', 'area', 'centroid', 'bbox'] 
props_dict = {"subject":[], "view":[], "slice_index":[], "index":[], "label":[], "area": [], "centroid-0": [], "centroid-1": [], "bbox-0": [], "bbox-1": [], "bbox-2": [], "bbox-3": [], "width":[], "height":[]} # index is only relevant if slice shows MULTIPLE nodules
views = ['top']

# Loop over and process all subjects
print("Creating {} slices for all CT scans.".format("& ".join(views)), flush=True)
for lab_file in all_labels:
    subject = lab_file[10:14]
    label_nifti = nib.load(label_inputpath+'/'+lab_file)
    label = label_nifti.get_fdata()
    label = label.astype(int) # to ensure integers, required by regionprops 
    label = skim.measure.label(label)
    lungs = np.load(f"{image_inputpath}/{subject}_lungs.npy")
    props_dict = create_slices(subject, lungs, label, properties, props_dict, datasplit, views, slice_outputpath)
print("Completed slicing all CT scans.", flush=True)    

# Save properties
props_table = pd.DataFrame(props_dict)
props_table.to_csv(props_outputpath)
print(f"Saved nodule properties for all slices to {props_outputpath}.", flush=True)


    
"""------------------------------------------ Step 4: B-Box Conversion -------------------------------------"""
from preprocessing_modules.yolo_formatting import to_yolo_format

# Access data
inputpath = "/home/u366836/DATA/LIDC/Nodule_properties_per_slice.csv" 
outputpath = "/home/u366836/DATA/LIDC/YOLO_Data_EXP1/labels"

# Loop over and process all subjects
print("Converting bounding boxes to YOLO format.", flush=True)
to_yolo_format(outputpath, inputpath, datasplit)
print("Completed YOLO bounding box conversion.", flush=True)



