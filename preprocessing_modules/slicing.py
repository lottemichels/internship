"""
Script to create 2D image slices of the nodules

The code below loops over all CT slices and extracts those that display a lesion.
Benign lesions or lesions < 3 mm are excluded here. 
The extracted CT slices, after lung mask application, are saved in an 'images' folder as .png files.
The nodule property data (area, bounding box, centroid, etc.) is saved in a 'properties.csv' file

"""

# Import packages
import os
import pandas as pd
import matplotlib.image
import skimage as skim # 3D image processing: 3D grayscale (plane, row, column)

def compute_properties(lab_slice, props_dict, subject, view, i, properties = ['label', 'area', 'centroid', 'bbox']):
    info_table = pd.DataFrame(skim.measure.regionprops_table(lab_slice, properties=properties)).reset_index()
    nodule_present = False
    nodule_amount = info_table.shape[0]
    labels = []
    if nodule_amount > 0: # there is a nodule in this slice!
        for nodule, props in info_table.iterrows():
            if props['bbox-2'] - props['bbox-0'] < 3 or props['bbox-3'] - props['bbox-1'] < 3: # the lesion width or height is very small: we skip these 
                continue
            nodule_present = True
            props_dict["subject"].append(subject)
            props_dict["view"].append(view) 
            props_dict["slice_index"].append(i)
            props_dict["height"].append(lab_slice.shape[0])
            props_dict["width"].append(lab_slice.shape[1])
            for prop, val in zip(props.keys(), props):
                props_dict[prop].append(val)
            labels.append(props['label'])
    return props_dict, nodule_present, labels


def save_slice(subject, view, i, lung_slice, datasplit, path, labels, singular=False): 
    # Datasplit determines in which set (train, val or test) a subject should be putted
    subject_num = int(subject)
    for split, (low, high) in datasplit.items():
        if low <= subject_num <= high:
            if not singular:
                matplotlib.image.imsave(f'{path}/{split}/{subject}_{view}_{i}.png', lung_slice, cmap='gray')
            elif singular:
                for lab in labels:
                    matplotlib.image.imsave(f'{path}/{split}/{subject}_{view}_{i}_n{int(lab)}.png', lung_slice, cmap='gray')
            break


def create_slices(subject, lungs, label, properties, props_dict, datasplit, views=['side','front','top'], path='/home/u366836/DATA/LIDC/YOLO_Data/images', singular=False):
    views_dims = {'side':0, 'front':1, 'top':2} # side = x = sagittal, front = y = coronal, top = z = axial
    dims = [views_dims[x] for x in views]
    for dim, view in zip(dims, views): 
        for i in range(0, label.shape[dim]): # loop over the indices to generate all possible slices 
            if dim == 0:
                lab_slice = label[i,:,:]
                lung_slice = lungs[i,:,:]
            elif dim == 1:
                lab_slice = label[:,i,:]
                lung_slice = lungs[:,i,:]
            elif dim == 2:
                lab_slice = label[:,:,i]
                lung_slice = lungs[:,:,i]
                
            # Compute regionprops for a given slice
            props_dict, nodule_present, labels = compute_properties(lab_slice, props_dict, subject, view, i, properties)
            
            # Save nodule containing lung slices
            if nodule_present:   
                save_slice(subject, view, i, lung_slice, datasplit, path, labels, singular)         
    return props_dict
