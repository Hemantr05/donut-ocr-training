import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import DonutProcessor

from utils import train_test_split

processor = DonutProcessor.from_pretrained('naver-clova-ix/donut-base-finetuned-cord-v2')


def loader(data, batch_size=16, shuffle=False, num_workers=0):
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

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