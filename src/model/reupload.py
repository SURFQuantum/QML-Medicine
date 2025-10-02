import torch
import torch.nn as nn
from torch.func import vmap 
import pennylane as qml
import pennylane.numpy as np

from .models import PCAMBackbone, TCGABackbone, ClassicalHead
from ..utils.algebra import compute_density_matrix

class QuantumHeadReupload(nn.Module):

    def __init__(self, 
                 input_shape: tuple,
                 num_classes: int, 
                 n_repetitions: int = 1, 
                 entangling_layer: str = 'strong'):
        
        super().__init__()
        self.n_qubits = input_shape[0]
        self.num_features_per_qubits = input_shape[1]
        self.n_input_blocs = -(len(self.num_features_per_qubits)//-3)
        self.num_classes = num_classes
        self.n_repetitions = n_repetitions 
        self.expected_latent_dim = input_shape
        self.num_params = 3 * self.n_qubits * self.n_repetitions
        self.q_params = nn.Parameter(torch.randn(self.n_qubits, 
                                                 3 * self.n_repetitions))
        self.etangling_layer = entangling_layer
        
        self.target_states = self.get_classes_state()
        self.target_density_matrices = torch.stack([compute_density_matrix(s) 
                                                    for s in self.target_states]) 
       
        dev = qml.device("default.qubit", wires=self.n_qubits)
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, q_params_, y):
            """A variational quantum circuit representing the Universal classifier.

            Args:
                params (array[float]): array of parameters
                x (array[float]): single input vector
                y (array[float]): single output state density matrix

            Returns:
                float: fidelity between output state and input
            """
            inputs = inputs.reshape(self.n_qubits,-1)
            for irep in range(self.n_repetitions):
                for iqubit in range(self.n_qubits):
                    for ibloc in range(self.n_input_blocs):
                        qml.Rot(*inputs[iqubit, ibloc*3:(ibloc+1)*3], wires=iqubit)
                    qml.Rot(*q_params_[iqubit, irep*3:(irep+1)*3], wires=iqubit)

                for iqubit in range(self.n_qubits):
                    ctrl = iqubit
                    target = (iqubit+1) % self.n_qubits
                    qml.CNOT(ctrl, target)
            return qml.expval(qml.Hermitian(y, wires=range(self.n_qubits)))     
        
        self.circuit = circuit

    def get_target_state(self):
        if self.num_classes == 2 :
            
            zeros = torch.zeros(2**self.n_qubits)
            zeros[0] = 1

            ones = torch.zeros(2**self.n_qubits)
            ones[-1] = 1

            return torch.stack((zeros, ones))
        else:
            raise ValueError("Reupload only works with two classes so far")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # input_dim = x.shape[1:]
        # if input_dim != self.expected_latent_dim:
        #     raise ValueError(
        #         f"Input dimension ({input_dim}) must match required latent dim "
        #         f"({self.expected_latent_dim}) for Reuploading."
        #     )
        dm_y = compute_density_matrix(y)
        return self.circuit(x, self.q_params, dm_y)

# =============================================================================
# Main Classifier
# =============================================================================
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
            num_classes = 1
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
            n_qubits = model_cfg['n_qubits']
            n_features_per_qubits = -(model_cfg["latent_dim"]//-n_qubits)

            if head_type == 'reupload':
                self.head = QuantumHeadReupload(
                    input_shape=(n_qubits, n_features_per_qubits),
                    num_classes=num_classes,
                    n_repetitions=model_cfg['n_quantum_layers'],
                    entangling_layer=model_cfg['entangling_layer']
                )
            else:
                raise ValueError(f"Unknown quantum_head_type: {head_type}")
        else:
            latent_dim = model_cfg['latent_dim']
            self.head = ClassicalHead(latent_dim, num_classes)
            
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x, y)