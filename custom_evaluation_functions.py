"""

Functions for custom DR and FPR metrics.

"""


# Import packages
from ultralytics import YOLO
import os
import shutil
import pandas as pd
import polars as pl


def detection_rate(model, version, test, custom_configuration, image_inputpath, label_inputpath, output_folder, conf=0.001):
    TP_dictionary = {'subject':[], 'nodule':[], 'subject_nodule':[], 'detected_at_least_once':[], 'nr_related_slices':[], 'nr_detections':[]}
    
    test_images = os.listdir(image_inputpath) 
    test_images.sort()
    test_labels = os.listdir(label_inputpath) # e.g. '0828_top_165.txt'
    test_labels.sort()

    # Define and initiate image placeholders
    image_placeholder = f'/home/u366836/DATA/LIDC/evaluation_placeholder_{version[0:7]}/images/test'
    label_placeholder = f'/home/u366836/DATA/LIDC/evaluation_placeholder_{version[0:7]}/labels/test'
    for image, label in zip(test_images, test_labels):
        # Re-instatiate the 'test' images and 'test' labels directories and copy in the current image and label
        shutil.rmtree(image_placeholder)
        os.makedirs(image_placeholder)
        shutil.copyfile(image_inputpath+'/'+image, image_placeholder+'/'+image)
        shutil.rmtree(label_placeholder)
        os.makedirs(label_placeholder)
        shutil.copyfile(label_inputpath+'/'+label, label_placeholder+'/'+label)
        # Extract subject and nodule number
        subject_nr = label[0:4]
        nidx = label.index('n')
        nodule_nr = label[nidx:nidx+2]
        subject_nodule = subject_nr+'_'+nodule_nr # e.g. 0128_n2
        # Run inference and extract results
        results = model.val(data=custom_configuration, split='test', project=output_folder+'/custom_runs', name=f'{version[0:7]}_{test}_evaluation_{subject_nodule}', conf=conf) # conf=0.25, iou=0.45
        TP = results.confusion_matrix.matrix[0,0]
        # Store results in dictionary
        if subject_nodule in TP_dictionary['subject_nodule']: # the nodule already came up before
            idx = TP_dictionary['subject_nodule'].index(subject_nodule)# get list index
            TP_dictionary['nr_related_slices'][idx] += 1
            if TP > 0:
                TP_dictionary['detected_at_least_once'][idx] = 1
                TP_dictionary['nr_detections'][idx] += 1
        else: # we haven't seen this nodule yet
            TP_dictionary['subject'].append(subject_nr)
            TP_dictionary['nodule'].append(nodule_nr)
            TP_dictionary['subject_nodule'].append(subject_nodule)
            TP_dictionary['detected_at_least_once'].append(1 if TP > 0 else 0)
            TP_dictionary['nr_related_slices'].append(1)
            TP_dictionary['nr_detections'].append(TP)  
    # Compute detection rate per nodule
    TP_dictionary['detection_rate'] = [x/y for x,y in zip(TP_dictionary['nr_detections'], TP_dictionary['nr_related_slices'])]
    
    # Save results to .csv file and output custom metric
    TP_dataframe = pd.DataFrame(TP_dictionary)
    TP_dataframe.to_csv(output_folder+f"/{version[0:7]}_{test}_{conf}_dr.csv")
    print(f"Saved custom detection rate results to {output_folder+f"/{version[0:7]}_{test}_{conf}_dr.csv"}.", flush=True)
    custom_dr = sum(TP_dictionary['detected_at_least_once'])/len(TP_dictionary['detected_at_least_once'])
    return custom_dr


def false_positive_rate(model, image_inputpath, conf=0.25):
    false_positive_images = []
    FP_counter = 0
    test_images = os.listdir(image_inputpath) 
    for image in test_images:
        # Run inference and extract results
        img_path = image_inputpath+'/'+image
        results = model.predict(img_path, conf=conf)
        for pred in results:
            nodule_detected = list(pred.boxes.cls)
            if nodule_detected:
                print('A false prediction was made for:', image, flush=True)
                false_positive_images.append(image) 
                FP_counter += 1
    custom_fpr = FP_counter / len(test_images)
    return false_positive_images, custom_fpr