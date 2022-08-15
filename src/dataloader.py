import os
import torch
import numpy as np
import multiprocessing as mp
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from transformers import DonutProcessor

processor = DonutProcessor.from_pretrained('naver-clova-ix/donut-base-finetuned-cord-v2')


def loader(trainset, testset, batch_size, shuffle=False, num_workers=0):
    if batch_size:
        if batch_size % mp.cpu_count() != 0:
            raise ValueError(f"Batch size {batch_size} must "
                            f"be divisible by device number {mp.cpu_count()}")
    train_loader = DataLoader(trainset, batch_size=batch_size, \
                                shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=1, \
                                shuffle=shuffle, num_workers=num_workers)

    return train_loader, test_loader

class DonutDataset(Dataset):
    def __init__(self, image_path, annotation_path, transforms=None):
        self.image_path = image_path
        self.annotation_path = annotation_path
        assert len(os.listdir(self.image_path)) == len(os.listdir(self.annotation_path))
        self.transforms = transforms

    def __len__(self):
        return len(os.listdir(self.annotation_path))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = os.path.join(self.image_path, os.listdir[idx])
        image = Image.open(path, 'L')
        sample = np.array(image)

        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def split(self, test_size=0.2):
        """Split dataset into train and test"""
        pass