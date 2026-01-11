"""
Functions to apply windowing and normalization

"""

# Import packages
import numpy as np

def apply_window(Input,W=1500,L=-600):
    """
    Window for lungs:
    W = 1500 and L = -600
    """
    min_HU=L-(0.5*W) # lower grey level 
    max_HU=L+(0.5*W) # upper grey level
    Input[Input<min_HU]=min_HU
    Input[Input>max_HU]=max_HU
    return Input

def apply_normalization(Input):
    minimum = Input.min()
    maximum = Input.max()
    Input = (Input - minimum) / (maximum - minimum)
    return Input
