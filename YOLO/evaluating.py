"""
Script to run standard and custom evaluations on YOLO models trained with the LIDC dataset
The YOLO version, weights and experiment to evaluate for should be provided.

"""

# Import packages
from ultralytics import YOLO
import os
import shutil
import pandas as pd
import polars as pl
import numpy as np
import json

from evaluation_modules.custom_evaluation_functions import detection_rate, false_positive_rate

# Set variables
weights = '/home/u366836/CODE/LIDC/YOLO/EXP3_runs_tuned/yolov5n_EXP3_train/weights/best.pt'
version = 'yolov5n.yaml' 
test = 'EXP3'
confs = [0.15, 0.25, 0.4, 0.5]
configuration = 'yolo_config.yaml'
custom_configuration = 'yolo_custom_evaluation_v5.yaml'
runs_folder = '/home/u366836/CODE/LIDC/YOLO/evaluation_runs'

# Initiate model
model = YOLO(weights)

# Get precision-recall curve
if os.path.exists(runs_folder+f'/{version[0:7]}_{test}_evaluation'):
  shutil.rmtree(runs_folder+f'/{version[0:7]}_{test}_evaluation')
results = model.val(data=configuration, split='test', project=runs_folder, name=f'{version[0:7]}_{test}_evaluation', visualize=True)
p = results.box.p_curve
p = p.ravel().tolist()
r = results.box.r_curve
r = r.ravel().tolist()
c = np.linspace(start=0, stop=1, num=1000)
c = c.ravel().tolist()
pr_curve = {'confidence':c, 'precision':p, 'recall':r}
pr_curve = pd.DataFrame(pr_curve)
pr_curve.to_csv(runs_folder+f'/{version[0:7]}_{test}_prcurve.csv')

# Placeholders for evaluation metrics
all_metrics = []
singular_image_inputpath = '/home/u366836/DATA/LIDC/all_singulars/images/test'
singular_label_inputpath = '/home/u366836/DATA/LIDC/all_singulars/labels/test'
test_false_image_inputpath = '/home/u366836/DATA/LIDC/test_negatives'
train_false_image_inputpath = '/home/u366836/DATA/LIDC/train_negatives'
for conf in confs:
  # Run standard evaluation (precision, recall, mAP) and custom evaluation (dr, fpr)
  # if os.path.exists(runs_folder+'/custom_runs'):
  #   shutil.rmtree(runs_folder+'/custom_runs')
  if os.path.exists(runs_folder+f'/{version[0:7]}_{test}_evaluation_{conf}'):
    shutil.rmtree(runs_folder+f'/{version[0:7]}_{test}_evaluation_{conf}')
  results = model.val(data=configuration, split='test', project=runs_folder, name=f'{version[0:7]}_{test}_evaluation_{conf}', conf=conf, visualize=True)
  custom_dr = detection_rate(model, version, test, custom_configuration, singular_image_inputpath, singular_label_inputpath, runs_folder, conf=conf)
  test_false_positive_images, test_custom_fpr = false_positive_rate(model, test_false_image_inputpath, conf=conf) 
  train_false_positive_images, train_custom_fpr = false_positive_rate(model, train_false_image_inputpath, conf=conf) 
  
  # Save metrics and false positive detections
  results_df = results.to_df()
  results_df = results_df.with_columns(pl.Series("custom_dr", [custom_dr]))
  results_df = results_df.with_columns(pl.Series("test_custom_fpr", [test_custom_fpr]))
  results_df = results_df.with_columns(pl.Series("train_custom_fpr", [train_custom_fpr]))
  results_df = results_df.with_columns(pl.Series("conf_threshold", [conf]))
  if isinstance(all_metrics, list):
    all_metrics = results_df
  else:
    all_metrics = pl.concat([all_metrics, results_df])
    
  json_filename = '/home/u366836/CODE/LIDC/YOLO/evaluation_runs/false_positives.json'
  if os.path.exists(json_filename):
      with open(json_filename, "r") as f:
          data = json.load(f)
  else:
    data = {}
  data[f"{version[0:7]}_{test}_{conf}_test"] = test_false_positive_images
  data[f"{version[0:7]}_{test}_{conf}_train"] = train_false_positive_images
  with open(json_filename, "w") as f:
    json.dump(data, f, indent=4)

all_metrics.write_csv(runs_folder+f"/{version[0:7]}_{test}_evaluation_metrics.csv")
