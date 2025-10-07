
from typing import Tuple
import torch
import torch.nn as nn

class TCGA(nn.Module):
    def __init__(self, input_dim: int = 768, latent_dim: int = 16):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Tanh()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)
    
class TCGASpatialPreserving(nn.Module):
    def __init__(self, input_dim: int = 768, latent_dim: int = 16,
                 out_grid: Tuple[int, int] = (4, 4)):
        """
        If preserve_spatial is False (default), matches original projection -> latent_dim.
        If preserve_spatial is True, we project to a vector of length H*W and return it
        (keeping a spatial layout for Hamiltonian construction). Caller must ensure
        latent_dim == H*W when using preserve_spatial=True.
        """
        super().__init__()
        self.out_grid = out_grid
        
        H, W = out_grid
        flat_dim = H * W
        self.projection = nn.Sequential(
            nn.Linear(input_dim, flat_dim),
            nn.LayerNorm(flat_dim),
            nn.ReLU(),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)