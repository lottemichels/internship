"""
Script to put bounding boxes of the image slices into YOLO format

YOLO format:
  1. folder called 'images' (optionally with 'train', 'validation', 'test' etc.) that contains:
      > your images (e.g. im1.png)
  2. folder called 'labels' (optionally with 'train', 'validation', 'test' etc.) that contains:
      > text files with congruent names (e.g. im1.txt). One line per object in the image states: 
          > the class number 
          > the bounding box: x center, y center, width, height

Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format 
where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

"""

# Import packages
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.image

def bbox_to_yolo_format(im_width, im_height, bbox_0, bbox_1, bbox_2, bbox_3):
    # bbox as returned by regionprops: (min_row, min_col, max_row, max_col)
    width, height, y_min, x_min, y_max, x_max = im_width, im_height, bbox_0, bbox_1, bbox_2, bbox_3
    x_center = ((x_min + x_max) / 2) / width
    y_center = ((y_min + y_max) / 2) / height
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height
    return [x_center, y_center, box_width, box_height]


def to_yolo_format(labelpath, propspath, datasplit, singular=False):
    # Re-instantiate label folder every time code is run 
    for split in datasplit.keys():
        shutil.rmtree(labelpath+"/"+split)
        os.makedirs(labelpath+"/"+split)

    props_df = pd.read_csv(propspath, dtype={'subject': str})
    for row, props in props_df.iterrows():
        # Get bounding box and other properties
        row, subject, view, slice_index, index, label, area, centroid_0, centroid_1, bbox_0, bbox_1, bbox_2, bbox_3, im_width, im_height = props.to_list()
        subject_num = int(subject)
        for split, (low, high) in datasplit.items():
            if low <= subject_num <= high:
                subject_split = split
                break
        
        # Convert bounding box
        x_center, y_center, box_width, box_height = bbox_to_yolo_format(im_width, im_height, bbox_0, bbox_1, bbox_2, bbox_3)
            
        # Save converted bounding box to .txt file
        txt_filename = '{}_{}_{}.txt'.format(subject, view, slice_index)
        txt_path = labelpath+"/"+subject_split+"/"
        if txt_filename in os.listdir(txt_path) and not singular: # If text file already exists (this slice contains multiple nodules): append 
            file = open(txt_path+txt_filename, "a")
            file.write("0 {} {} {} {} \n".format(x_center, y_center, box_width, box_height)) # 0 = class label (nodule)
            file.close()
        elif singular:
            txt_filename = '{}_{}_{}_n{}.txt'.format(subject, view, slice_index, int(label))
            file = open(txt_path+txt_filename , "w")
            file.write("0 {} {} {} {} \n".format(x_center, y_center, box_width, box_height)) # 
            file.close()
        else: # Else create a new text file 
            file = open(txt_path+txt_filename , "w")
            file.write("0 {} {} {} {} \n".format(x_center, y_center, box_width, box_height)) # 
            file.close()