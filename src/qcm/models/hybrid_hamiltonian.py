# model/models.py

import torch
import torch.nn as nn


from ..components.encoder.pcam import PCAMSpatialPreserving as PCAMBackbone
from ..components.encoder.tcga import TCGASpatialPreserving as TCGABackbone

from ..components.hamiltonian.embedding import QuantumHeadHamiltonianPaper
from ..components.hamiltonian.embedding_light import QuantumHeadHamiltonianSimple


from ..utils.image_grid import grid_shape_for_amplitude

# =============================================================================
# Main Classifier (only applies spatial-preserving backbone when head_type == 'hamiltonian')
# =============================================================================
class HybridClassifier(nn.Module):
    def __init__(self, config: dict, use_quantum: bool = False):
        super().__init__()
        
        model_cfg = config['model']
        dataset_type = config['dataset_type']
        n_qubits = model_cfg['n_qubits']
        num_classes = model_cfg['num_classes']

        latent_dim = 2 ** n_qubits
        grid_height, grid_width = grid_shape_for_amplitude(n_qubits)

        if dataset_type == 'tcga':
            self.backbone = TCGABackbone(input_dim=768, 
                                         latent_dim=latent_dim,
                                         out_grid=(grid_height, grid_width))
        else:
            self.backbone = PCAMBackbone(latent_dim=latent_dim,
                                         filters=model_cfg['pcam_filters'],
                                         out_grid=(grid_height, grid_width))
            
        variant = model_cfg.get('hamiltonian_variant', 'simple')  # 'simple' or 'paper'
        if variant == 'simple':
            self.head = QuantumHeadHamiltonianSimple(
                n_qubits=n_qubits,
                num_classes=num_classes,
                n_layers=model_cfg['n_quantum_layers'],
                entangling_layer=model_cfg['entangling_layer'],
                include_zz=model_cfg.get('hamiltonian_include_zz', True),
                time=model_cfg.get('hamiltonian_time', 1.0)
            )
            
        elif variant == 'paper':
            self.head = QuantumHeadHamiltonianPaper(
                n_qubits=n_qubits,
                num_classes=num_classes,
                n_layers=model_cfg['n_quantum_layers'],
                entangling_layer=model_cfg['entangling_layer'],
                include_xx=model_cfg.get('hamiltonian_include_xx', True),
                n_trotter_steps=model_cfg.get('hamiltonian_trotter_steps', 1),
                time=model_cfg.get('hamiltonian_time', 1.0)
            )
        else:
            raise ValueError(f"Unknown hamiltonian_variant: {variant}")
            

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x)