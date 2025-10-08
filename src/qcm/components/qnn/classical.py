
import torch
import torch.nn as nn

class ClassicalHead(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int):
        super().__init__()
        self.out = nn.Linear(latent_dim, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(x)
