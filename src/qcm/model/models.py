# model/models.py

import torch
import torch.nn as nn
from torch.func import vmap 
import pennylane as qml

from ..backbone.PCAM import PCAM as PCAMBackbone
from ..backbone.TCGA import TCGA as TCGABackbone

# =============================================================================
# Heads
# =============================================================================
class ClassicalHead(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int):
        super().__init__()
        self.out = nn.Linear(latent_dim, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(x)

class QuantumHeadAngle(nn.Module):
    def __init__(self, n_qubits: int, num_classes: int, n_layers: int = 1, entangling_layer: str = 'strong'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        if entangling_layer == 'strong':
            self.q_params = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
            entangler = qml.templates.StronglyEntanglingLayers
        elif entangling_layer == 'basic':
            self.q_params = nn.Parameter(torch.randn(n_layers, n_qubits))
            entangler = qml.templates.BasicEntanglerLayers
        else:
            raise ValueError(f"Unknown entangling_layer: {entangling_layer}")

        dev = qml.device("default.qubit", wires=n_qubits)
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, q_params_):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
            entangler(q_params_, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        self.circuit = circuit
        self.classifier = nn.Linear(n_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def circuit_wrapper(single_input):
            return self.circuit(single_input, self.q_params)
        
        raw_vmap_output = vmap(circuit_wrapper)(x)
        quantum_features = torch.stack(raw_vmap_output, dim=1).float()
        
        return self.classifier(quantum_features)

class QuantumHeadAmplitude(nn.Module):
    def __init__(self, n_qubits: int, num_classes: int, n_layers: int = 1, entangling_layer: str = 'strong'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.required_latent_dim = 2**n_qubits

        if entangling_layer == 'strong':
            self.q_params = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
            entangler = qml.templates.StronglyEntanglingLayers
        elif entangling_layer == 'basic':
            self.q_params = nn.Parameter(torch.randn(n_layers, n_qubits))
            entangler = qml.templates.BasicEntanglerLayers
        else:
            raise ValueError(f"Unknown entangling_layer: {entangling_layer}")

        dev = qml.device("default.qubit", wires=n_qubits)
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, q_params_):
            qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True)
            entangler(q_params_, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        self.circuit = circuit
        self.classifier = nn.Linear(n_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, input_dim = x.shape
        if input_dim != self.required_latent_dim:
            raise ValueError(
                f"Input dimension ({input_dim}) must match required latent dim "
                f"({self.required_latent_dim} = 2^{self.n_qubits}) for AmplitudeEmbedding."
            )
        
        # === FIX IS HERE ===
        # 1. raw_features is a LIST of tensors, one for each measurement.
        raw_features = self.circuit(x, self.q_params)
        # 2. Stack the list into a single tensor. Shape becomes (n_qubits, batch_size)
        stacked_features = torch.stack(raw_features, dim=0)
        # 3. Permute to (batch_size, n_qubits) and cast to float for the linear layer.
        quantum_features = stacked_features.permute(1, 0).float()
        
        return self.classifier(quantum_features)

# =============================================================================
# Main Classifier
# =============================================================================
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