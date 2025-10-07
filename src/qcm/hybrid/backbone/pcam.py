import torch
import torch.nn as nn

# =============================================================================
# Backbones
# =============================================================================
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