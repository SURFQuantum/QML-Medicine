import torch
import torch.nn as nn
from typing import Tuple

class PCAM(nn.Module):

    def __init__(self, latent_dim: int = 64, filters: int = 4):
        super().__init__()
        multiplier = 2
        self.features = nn.Sequential(
            nn.Conv2d(3, filters, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters), nn.ReLU(),
            nn.Conv2d(filters, filters*multiplier, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters*multiplier), nn.ReLU(),
            nn.Conv2d(filters*multiplier, filters*multiplier**2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters*multiplier**2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.latent = nn.Sequential(
            nn.Linear(filters*multiplier**2 * 2 * 2, latent_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.latent(x)
    

# =============================================================================
# Backbones (modified to support optional spatial-preserving output)
# =============================================================================
class PCAMSpatialPreserving(nn.Module):
    def __init__(self, latent_dim: int = 64, filters: int = 4,
                 out_grid: Tuple[int, int] = (2, 2)):
        """
        If preserve_spatial is False (default), behavior matches the original:
          - AdaptiveAvgPool2d((2,2)), flatten, then Linear -> latent_dim.
        If preserve_spatial is True, we:
          - AdaptiveAvgPool2d(out_grid), apply a 1x1 conv to produce one scalar per cell,
            and return the flattened grid of length H*W (no final dense mixing).
        This lets the caller control whether the backbone preserves 2D locality.
        """
        super().__init__()
        multiplier = 2
        self.out_grid = out_grid

        self.features = nn.Sequential(
            nn.Conv2d(3, filters, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters), nn.ReLU(),
            nn.Conv2d(filters, filters*multiplier, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters*multiplier), nn.ReLU(),
            nn.Conv2d(filters*multiplier, filters*multiplier**2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters*multiplier**2), nn.ReLU()
        )
        
        H, W = out_grid
        # Replace the final pooling with one that outputs the desired grid
        self.pool = nn.AdaptiveAvgPool2d((H, W))
        # project channel dimension to 1 per grid cell (a scalar per cell)
        self.cell_proj = nn.Conv2d(filters*multiplier**2, 1, kernel_size=1)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)    
        x = self.cell_proj(x)         # (B,1,H,W)
        x = x.view(x.size(0), -1)     # (B, H*W)
        return x
