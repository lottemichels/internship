"""
For each nodule in the LIDC dataset, create slices containing that nodule and that nodule ONLY.
This is needed for the custom detection rate metric, where we check for each nodule whether is has been detected in one of its slices or not.

"""
# Import packages
import os
import shutil
import numpy as np
import pandas as pd
import nibabel as nib # NIFTI file processing
import skimage as skim # 3D image processing: 3D grayscale (plane, row, column)

# Defining global variables
datasplit = {"train":(1, 660), "val":(661, 827), "test":(828, 1012)}
singular=True


# Slicing
from preprocessing_modules.slicing import create_slices 

# Access data
image_inputpath = "/home/u366836/DATA/LIDC/Masked_Scans_EXP1" 
label_inputpath = "/home/mleeuwen/DATA/LIDC_data/New_Uniform_labels_nodules_ambiguous"
slice_outputpath = "/home/u366836/DATA/LIDC/evaluation_folder/images"
props_outputpath = "/home/u366836/DATA/LIDC/Nodule_properties_per_singular_slice.csv" # this remains constant between experiments

all_labels = os.listdir(label_inputpath)
all_labels.sort()
print(f"Found {len(os.listdir(image_inputpath))} lungfiles and {len(all_labels)} labels.")

# Re-instantiate image folder every time code is run 
for split in datasplit.keys():
    shutil.rmtree(slice_outputpath+"/"+split)
    os.makedirs(slice_outputpath+"/"+split)

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
    props_dict = create_slices(subject, lungs, label, properties, props_dict, datasplit, views, slice_outputpath, singular=singular)
print("Completed slicing all CT scans.", flush=True)    

# Save properties
props_table = pd.DataFrame(props_dict)
props_table.to_csv(props_outputpath)
print(f"Saved nodule properties for all slices to {props_outputpath}.", flush=True)

# YOLO Formatting
from preprocessing_modules.yolo_formatting import to_yolo_format

# Access data
inputpath = "/home/u366836/DATA/LIDC/Nodule_properties_per_singular_slice.csv" 
outputpath = "/home/u366836/DATA/LIDC/all_singulars/labels"

# Loop over and process all subjects
print("Converting bounding boxes to YOLO format.", flush=True)
to_yolo_format(outputpath, inputpath, datasplit, singular=singular)
print("Completed YOLO bounding box conversion.", flush=True)