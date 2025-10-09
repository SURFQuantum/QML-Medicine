import torch
from torch.utils.data import Dataset

# =============================================================================
# Horizontal and vertical bands Dataset (Used for testing)
# =============================================================================
class BandDataset(Dataset):
    def __init__(self, size: int = 16, num_samples: int = 1000, max_filling = 0.75):
        """A data set composed of vertical and horizontal bands.

        Args:
            size (int): number of pixels for the width and height of the image
            num_samples (int): number of samples in the dataset
        """
        self.size = size
        self.num_samples = num_samples

        self.num_min_band = 1
        self.num_max_band = int(size * max_filling)

        self.indices = self._get_indices()

    def _get_indices(self):
        indices = []
        for _ in range(self.num_samples):
            num_bands = torch.randint(self.num_min_band, self.num_max_band, (1,)).item()
            indices.append(torch.randint(0, self.size, (num_bands,)))
        return indices

    def __len__(self):
        return len(self.num_samples)

    def __getitem__(self, idx):

        image = torch.zeros(self.size, self.size)
        label = torch.randint(2, (1,)).item()

        index = self.indices[idx]

        if label == 0:
            image[index, :] = 1.0
        else:
            image[:, index] = 1.0
            
        return image, torch.tensor(label).float()