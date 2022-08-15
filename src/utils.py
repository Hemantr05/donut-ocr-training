import os
from torchvision import transforms, utils

def save_model():
    """
    Apply model versioning
    Version based on `run_name-time`
    
    """
    pass

def load_model():
    pass

def train_test_split(annotations, images, test_split=0.2):
    pass

def fetch_transforms(crop_size:int):
    """Reference: https://pytorch.org/vision/0.9/transforms.html"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(),
        transforms.CenterCrop(crop_size),
    ])