import torch
from torch.utils.data import Dataset

# =============================================================================
# Horizontal and vertical bands Dataset (Used for testing)
# =============================================================================
class BandDataset(Dataset):
    def __init__(self, 
                 image_size: int = 16, 
                 num_samples: int = 1000, 
                 max_filling = 0.75,
                 num_channels: int = 1):
        """A data set composed of vertical and horizontal bands.

        Args:
            size (int): number of pixels for the width and height of the image
            num_samples (int): number of samples in the dataset
        """
        self.size = image_size
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.num_min_band = 1
        self.num_max_band = int(image_size * max_filling)

        self.indices = self._get_indices()

    def _get_indices(self):
        indices = []
        for _ in range(self.num_samples):
            num_bands = torch.randint(self.num_min_band, self.num_max_band, (1,)).item()
            indices.append(torch.randint(0, self.size, (num_bands,)))
        return indices

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        image = torch.zeros(self.num_channels, self.size, self.size)
        label = torch.randint(2, (1,)).item()
        if self.num_channels == 1:
            val = 1.0
        else:
            val = torch.rand(1)
        index = self.indices[idx]

        if label == 0:
            image[:, index, :] = val
        else:
            image[:, :, index] = val
            
        return image, torch.tensor(label).float()