"""
Script to run a YOLO model.

The code works with three variables:
- versions: the YOLO version you want to tune
- test: which experiment to tune for
- tuning: whether to run an untuned model or use tuned hyperparameters. In the latter case, a path to the hyperparameters should be defined as well.

"""


# Variables
version = 'yolov5n.yaml'
test = 'EXP3' # EXP1 or EXP2
tuning = 'tuned' # 'tuned' or 'untuned'
hyperparams = '/home/u366836/CODE/LIDC/YOLO/runs/detect/tune_yolov5n_EXP3/best_hyperparameters.yaml' # path to tuned hyperparameters, or None for untuned
vn = version[0:7] # version number
configuration = f'yolo_config_{test}.yaml'

# Import packages
from ultralytics import YOLO

# Initiate
model = YOLO(version)

# Train 
if test == 'EXP1':
    if tuning == 'tuned':
        train_results = model.train(data=configuration, cfg=hyperparams, project=f'EXP1_runs_{tuning}', epochs=200, split='train', name=f'{vn}_{test}_train')
    else:
        train_results = model.train(data=configuration, project=f'EXP1_runs_{tuning}', epochs=200, split='train', name=f'{vn}_{test}_train')

elif test == 'EXP2':
    if tuning == 'tuned':
        train_results = model.train(data=configuration, cfg=hyperparams, project=f'EXP2_runs_{tuning}', epochs=200, split='train', name=f'{vn}_{test}_train')  
    else:
        train_results = model.train(data=configuration, project=f'EXP2_runs_{tuning}', epochs=200, split='train', name=f'{vn}_{test}_train')  

elif test == 'EXP3':
    if tuning == 'tuned':
        train_results = model.train(data=configuration, cfg=hyperparams, project=f'EXP3_runs_{tuning}', epochs=200, split='train', name=f'{vn}_{test}_train')  
    else:
        train_results = model.train(data=configuration, project=f'EXP3_runs_{tuning}', epochs=200, split='train', name=f'{vn}_{test}_train')  
            