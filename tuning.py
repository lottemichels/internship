"""
Script to tune the YOLO models

The code works with two variables:
- versions: the YOLO version you want to tune
- test: which experiment to tune for. This variable also determines the tuning configuration file and hyperparameter search space

"""

# Variables
version = 'yolov5n.yaml'
test = 'EXP2'
configuration = f'yolo_config_{test}.yaml'

if test == 'EXP1':
    search_space = {
        "lr0": (1e-5, 1e-1), # def = 0.01
        "lrf": (0.01, 1.0), 
        "box": (0.02, 7.5), # def = 7.5
        "cls": (0.2, 1.0), # def = 0.5
        "weight_decay": (0.0, 0.001), # def = 0.0005
        "momentum": (0.8, 0.98), # def = 0.937
        "warmup_epochs": (0.0, 5.0), # def = 3.0
        "warmup_momentum": (0.0, 0.95), # def = 0.8
    }
else:
    search_space = {
        "lr0": (1e-5, 1e-1), # def = 0.01
        "lrf": (0.01, 1.0), 
        "box": (0.02, 7.5), # def = 7.5
        "cls": (0.2, 1.0), # def = 0.5
        "weight_decay": (0.0, 0.001), # def = 0.0005
        "momentum": (0.8, 0.98), # def = 0.937
        "warmup_epochs": (0.0, 5.0), # def = 3.0
        "warmup_momentum": (0.0, 0.95), # def = 0.8
        "degrees": (0, 20),
        "translate": (0.0, 0.25),
        "scale": (0.0, 0.4),
        "fliplr": (0, 1)
    }
      
    
# Import packages
from ultralytics import YOLO

# Initiate the model
model = YOLO(version)
# Train and tune
results = model.tune(data=configuration, split='train', epochs=75, iterations=100, name=f"tune_{version[0:7]}_{test}", resume=True)
