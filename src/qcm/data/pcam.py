import os
import h5py
import torch

import numpy as np
from torch.utils.data import Dataset

# =============================================================================
# PCAM Dataset (Original)
# =============================================================================
class PCAMDataset(Dataset):
    def __init__(self, x_path: str, y_path: str, transform=None):
        self.x_path = x_path
        self.y_path = y_path
        self.transform = transform

        self.x_data = h5py.File(x_path, 'r')['x']
        self.y_data = h5py.File(y_path, 'r')['y']

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        image = self.x_data[idx]
        label = self.y_data[idx][0]
        image = image.astype(np.uint8)

        if self.transform:
            image = self.transform(image)
            
        # PCAM is binary classification, requires float for BCEWithLogitsLoss
        return image, torch.tensor(label).float()