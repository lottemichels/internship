"""
Script to add samples of negative samples to the training and validation set.

"""

# Import packages
from ultralytics import YOLO
import os
import shutil
import pandas as pd
import polars as pl
import numpy as np
import json
import random

random.seed(42) # to select the same sample every run

# Augment training datasets

# Get hard negatives to create a sampling pool
json_filename = '/home/u366836/CODE/LIDC/YOLO/evaluation_runs/false_positives.json'
with open(json_filename, "r") as f:
        data = json.load(f)
v5_false_train = data['yolov8n_EXP1_0.5_train'] # I use 0.5 conf threshold because these are the negative outputs the model is most confident about (i.e. the hardest negative samples)
v8_false_train = data['yolov5n_EXP1_0.5_train']
v11_false_train = data['yolo11n_EXP1_0.5_train']
pool = list(set(v5_false_train) & set(v8_false_train) & set(v11_false_train))
pool_len = len(pool)
print('The negative sample pool for training data is of size:', pool_len, flush=True)

# Get training set magnitudes to determine false negative ratios
train_path = '/home/u366836/DATA/LIDC/YOLO_Data_EXP1/images/train'
train_data = os.listdir(train_path)
train_len = len(train_data)
print('The original training set is of size:', train_len, flush=True)

ratio1 = 0.10 # 10%
train_neg_sample_len1 = int(ratio1 * train_len)
print('A ratio of 0.1 generates a sample of size:', train_neg_sample_len1, flush=True)
ratio2 = 0.20 # 20%
train_neg_sample_len2 = int(ratio2 * train_len)
print('A ratio of 0.2 generates a sample of size:', train_neg_sample_len2, flush=True)

# Sample negative slices
negative_train_sample1 = random.sample(pool, train_neg_sample_len1)
print('Check:', len(negative_train_sample1), flush=True)
negative_train_sample2 = random.sample(pool, train_neg_sample_len2)
print('Check:', len(negative_train_sample2), flush=True)

# Augment training datasets 
for n in negative_train_sample1:
    n_file = f'/home/u366836/DATA/LIDC/train_negatives/{n}'
    shutil.copyfile(n_file, f'/home/u366836/DATA/LIDC/YOLO_Data_EXP2/images/train/{n}')
    with open(f'/home/u366836/DATA/LIDC/YOLO_Data_EXP2/labels/train/{n.strip('.png')}.txt', 'w') as fp:
        pass
print('Augmented EXP2 training data', flush=True)
for n in negative_train_sample2:
    n_file = f'/home/u366836/DATA/LIDC/train_negatives/{n}'
    shutil.copyfile(n_file, f'/home/u366836/DATA/LIDC/YOLO_Data_EXP3/images/train/{n}')
    with open(f'/home/u366836/DATA/LIDC/YOLO_Data_EXP3/labels/train/{n.strip('.png')}.txt', 'w') as fp:
        pass
print('Augmented EXP3 training data', flush=True)

