from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class BandEncoder(nn.Module):

    def __init__(self, filters: int = 6, latent_dim: int = 8):
        """Encoder for the band dataset.

        This only works if the input images are 16x16.

        Args:
            filters (int, optional): number of filter in the first cnn. Defaults to 6.
            latent_dim (int, optional): size of the output vector. Defaults to 8.
        """
        multipliers = 2
        super().__init__()
        self.conv1 = nn.Conv2d(1, filters, kernel_size = 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(filters, filters*multipliers, kernel_size=3)
        self.fc1 = nn.Linear(48, latent_dim)
    

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Input must be a 4D tensor (batch_size, channels, height, width)")
        if x.shape[-1] != 16 or x.shape[-2] != 16:
            raise ValueError("Input images must be 16x16")
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.tanh(self.fc1(x))
        
        return x