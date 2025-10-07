
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