import torch
from torch.utils.data import Dataset

# =============================================================================
# Horizontal and vertical bands Dataset (Used for testing)
# =============================================================================
class BandDataset(Dataset):
    def __init__(self, size: int, num_samples: int):
        """A data set composed of vertical and horizontal bands.

        Args:
            size (int): number of pixels for the width and height of the image
            num_samples (int): number of samples in the dataset
        """
        self.size = size
        self.num_samples = num_samples

        self.num_min_band = 1
        self.num_max_band = 2 * self.size // 3

    def __len__(self):
        return len(self.num_samples)

    def __getitem__(self, idx):

        image = torch.zeros(self.size, self.size)
        label = torch.randint(2, (1,)).item()

        num_bands = torch.randint(self.num_min_band, self.num_max_band, (1,)).item()
        idx = torch.randint(0, self.size, (num_bands,))

        if label == 0:
            image[idx, :] = 1.0
        else:
            image[:, idx] = 1.0
            
        return image, torch.tensor(label).float()