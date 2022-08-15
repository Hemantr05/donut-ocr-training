import os
import torch
from torchvision import transforms

from cloud_storage import model_to_bucket

def save_model(model, run_name, epoch, cloud_store=False, metrics_required=False):
    """
    Apply model versioning
    Version based on `run_name-time`
    """
    ckpt_name = f'{str(epoch)}.pt'


    parent, child = tuple(run_name.split('-'))
    os.makedirs(parent, exist_ok=True)
    path = os.path.join(os.getcwd(), parent)
    os.makedirs(child, exist_ok=True)
    path = os.path.join(path, child)
    path = os.path.join(path, ckpt_name)

    torch.save(model.state_dict(), path)

    if cloud_store:
        model_to_bucket(parent, child, path)


def load_model():
    pass

def get_transforms():
    """Reference: https://pytorch.org/vision/0.9/transforms.html"""
    transform = transforms.Compose([
        # transforms.RandomCrop(crop_size),
        # transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
    ])

    return transform