import torch
from torch.utils.data import Dataset
import numpy as np 

class FixedLatentSpace(Dataset):
    """This datasset contains precomputed latent space representations.

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, x_path: str, y_path: str, num_features: int = None, num_samples: int = None):
        self.x_path = x_path
        self.y_path = y_path
    
        self.x_data = np.load(x_path)
        self.y_data = np.load(y_path)

        if num_samples is not None:
            self.x_data = self.x_data[:num_samples]
            self.y_data = self.y_data[:num_samples]

        if num_samples is not None:
            self.x_data = self.x_data[:, :num_features]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        latent_features = self.x_data[idx]
        label = self.y_data[idx]
 
        return torch.tensor(latent_features), torch.tensor(label).float()