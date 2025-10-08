# model/models.py

import torch
import torch.nn as nn


from ..components.encoder.pcam import PCAM as PCAMBackbone
from ..components.encoder.tcga import TCGA as TCGABackbone


from ..components.qnn.classical import ClassicalHead
from ..components.qnn.amplitude_encoding import QuantumHeadAmplitude
from ..components.qnn.angle_encoding import QuantumHeadAngle


class HybridClassifier(nn.Module):
    def __init__(self, config: dict, use_quantum: bool = False, num_quantum_layers: int = 1):
        super().__init__()
        model_cfg = config['model']
        dataset_type = config['dataset_type']
        self.num_quantum_layers = num_quantum_layers
        
        # 1. Select Backbone
        if dataset_type == 'pcam':
            self.backbone = PCAMBackbone(
                latent_dim=model_cfg['latent_dim'], 
                filters=model_cfg['pcam_filters']
            )
            num_classes = 1
        elif dataset_type == 'tcga':
            self.backbone = TCGABackbone(
                input_dim=768, 
                latent_dim=model_cfg['latent_dim']
            )
            num_classes = model_cfg['num_classes']
        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

        # 2. Select Head
        if use_quantum:
            head_type = model_cfg.get('quantum_head_type', 'amplitude')
            n_qubits = model_cfg['n_qubits']
            
            if head_type == 'amplitude':
                latent_dim = 2**n_qubits
                if dataset_type == 'tcga':
                     self.backbone = TCGABackbone(input_dim=768, latent_dim=latent_dim)
                else:
                     self.backbone = PCAMBackbone(latent_dim=latent_dim, filters=model_cfg['pcam_filters'])

                self.head = QuantumHeadAmplitude(
                    n_qubits=n_qubits,
                    num_classes=num_classes,
                    n_layers=self.num_quantum_layers,
                    entangling_layer=model_cfg['entangling_layer']
                )
            elif head_type == 'angle':
                latent_dim = n_qubits
                if dataset_type == 'tcga':
                     self.backbone = TCGABackbone(input_dim=768, latent_dim=latent_dim)
                else:
                     self.backbone = PCAMBackbone(latent_dim=latent_dim, filters=model_cfg['pcam_filters'])

                self.head = QuantumHeadAngle(
                    n_qubits=n_qubits,
                    num_classes=num_classes,
                    n_layers=self.num_quantum_layers,
                    entangling_layer=model_cfg['entangling_layer']
                )
            else:
                raise ValueError(f"Unknown quantum_head_type: {head_type}")
        else:
            latent_dim = model_cfg['latent_dim']
            self.head = ClassicalHead(latent_dim, num_classes)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x)