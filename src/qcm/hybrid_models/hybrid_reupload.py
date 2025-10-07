import torch
import torch.nn as nn
from ..encoder.pcam import PCAM as PCAMBackbone
from ..encoder.tcga import TCGA as TCGABackbone
from ..qnn.classical import ClassicalHead
from ..reupload.reupload import QuantumHeadReupload

class HybridReuploadClassifier(nn.Module):
    def __init__(self, config: dict, use_quantum: bool = False):
        super().__init__()
        model_cfg = config['model']
        dataset_type = config['dataset_type']
        
        # 1. Select Backbone
        if dataset_type == 'pcam':
            self.backbone = PCAMBackbone(
                latent_dim=model_cfg['latent_dim'], 
                filters=model_cfg['pcam_filters']
            )
            num_classes = 2
        elif dataset_type == 'tcga':
            self.backbone = TCGABackbone(
                input_dim = 768, 
                latent_dim=model_cfg['latent_dim']
            )
            num_classes = model_cfg['num_classes']
        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

        # 2. Select Head
        if use_quantum:
            head_type = model_cfg.get('quantum_head_type', 'reupload')

            if head_type == 'reupload':
                self.head = QuantumHeadReupload(
                    num_qubits = model_cfg['n_qubits'],
                    num_features = model_cfg['latent_dim'],
                    num_classes=num_classes,
                    n_repetitions=model_cfg['n_quantum_layers'],
                    entangling_layer=model_cfg['entangling_layer']
                )
            else:
                raise ValueError(f"Unknown quantum_head_type: {head_type}")
        else:
            latent_dim = model_cfg['latent_dim']
            self.head = ClassicalHead(latent_dim, num_classes)
    
        self.normalisation = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.normalisation(x)
        return self.head(x, y)
    
class ReuploadClassifier(nn.Module):
    def __init__(self, config: dict, use_quantum: bool = False):
        super().__init__()
        model_cfg = config['model']

        num_classes = 2

        self.head = QuantumHeadReupload(
            num_qubits = model_cfg['n_qubits'], 
            num_features = model_cfg['latent_dim'],
            num_classes=num_classes,
            n_repetitions=model_cfg['n_quantum_layers'],
            entangling_layer=model_cfg['entangling_layer']
        )    

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.head(x, y)