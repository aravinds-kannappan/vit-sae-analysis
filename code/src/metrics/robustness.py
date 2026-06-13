import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from main.prep_data import prep_data

def evaluate_robustness_jpeg(model, dataset, RPI = False, magnitude = 1.0):
    # Prepare dataloaders for the JPEG-corrupted and normal datasets
    
    dataloader_jpeg = prep_data(dataset, model.config, "JPEG")
    dataloader_normal = prep_data(dataset, model.config, None)

    # Evaluate accuracy on both procured datasets

    mean_acc = model.predict(dataloader_normal, RPI, magnitude)
    jpeg_acc = model.predict(dataloader_jpeg, RPI, magnitude)

    return (1 - (jpeg_acc/mean_acc))  # Evaluate and return fragility score

def evaluate_robustness_gaussian_blur(model, dataset, RPI = False, magnitude = 1.0):
    # Prepare dataloaders for the Gaussian-blur-corrupted and normal datasets

    dataloader_GB = prep_data(dataset, model.config, "Gaussian Blur")
    dataloader_normal = prep_data(dataset, model.config, None)

    # Evaluate accuracy on both procured datasets
    mean_acc = model.predict(dataloader_normal, RPI, magnitude)
    GB_acc = model.predict(dataloader_GB, RPI, magnitude)

    return (1 - (GB_acc /mean_acc)) # Evaluate and return fragility score