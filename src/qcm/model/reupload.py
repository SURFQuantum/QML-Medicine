from typing import List
import torch
import torch.nn as nn
from torch.func import vmap 
import pennylane as qml
import pennylane.numpy as np
from .models import PCAMBackbone, TCGABackbone, ClassicalHead
from ..utils.algebra import compute_density_matrix


class QuantumHeadReupload(nn.Module):

    def __init__(self, 
                 num_qubits: int,
                 num_features: int,
                 num_classes: int, 
                 n_repetitions: int = 1, 
                 entangling_layer: str = 'strong'):
        
        super().__init__()

        self.num_qubits = num_qubits
        self.num_features = num_features

        self.num_features_per_qubits = -(self.num_features // -self.num_qubits)
        self.n_input_blocs = -(self.num_features_per_qubits//-3)
        
        self.num_classes = num_classes
        self.n_repetitions = n_repetitions 

        self.num_params = 3 * self.num_qubits * self.n_repetitions
        self.q_params = nn.Parameter(torch.randn(self.num_qubits, 
                                                 3 * self.n_repetitions))
        self.etangling_layer = entangling_layer
        
        self.target_labels = self.get_target_labels()
        self.target_states = self.get_target_states()
        self.target_density_matrices = torch.stack([compute_density_matrix(s) 
                                                    for s in self.target_states]) 
    
        self.input_pad_size = self.num_qubits * self.num_features_per_qubits - self.num_features
        self.pad_layer = torch.nn.ZeroPad1d((0, self.input_pad_size))

        dev = qml.device("default.qubit", wires=self.num_qubits)
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor, q_params_: torch.Tensor, y: torch.Tensor) -> float:
            """Quantum Circuit of the head.

            Args:
                inputs (torch.tensor): the input data to encode in the rotation of the gates
                q_params_ (torch.tensor): variational parameters of the circuit
                y (torch.tensor): density matrix associated with the target label

            Returns:
                float: fidelity of the circuit w.r.t the target density matrix
            """
            inputs = inputs.reshape(self.num_qubits,-1)
            for irep in range(self.n_repetitions):
                for iqubit in range(self.num_qubits):
                    for ibloc in range(self.n_input_blocs):
                        qml.Rot(*inputs[iqubit, ibloc*3:(ibloc+1)*3], wires=iqubit)
                    qml.Rot(*q_params_[iqubit, irep*3:(irep+1)*3], wires=iqubit)

                if self.num_qubits > 1:
                    for iqubit in range(self.num_qubits):
                        ctrl = iqubit
                        target = (iqubit+1) % self.num_qubits
                        qml.CNOT(wires=[ctrl, target])
            return qml.expval(qml.Hermitian(y, wires=range(self.num_qubits)))     
        
        self.circuit = circuit

    def get_pad_size(self) -> int:
        """Get the padding size for the input data."""
        mult = 3 * self.num_qubits
        return 3 - self.num_features % mult        

    def get_target_labels(self) -> torch.Tensor:
        """Generates the possible labels of the dataset

        Returns:
            torch.Tensor: tensor containing the possible values
        """
        return torch.tensor(range(self.num_classes)).reshape(-1,1)
       

    def get_target_states(self) -> torch.Tensor:
        """Computes the target states associtated with the different labels

        Raises:
            ValueError: _description_

        Returns:
            torch.Tensor: target states
        """
        if self.num_classes > 2**self.num_qubits :
            raise ValueError(f"The number of possible output state in the computational basis: {2**self.num_qubits}" \
            f"is smaller than the number of classes in the problem: {self.num_classes}" \
            "Increase the number of qubits in the circuits or reduce the number classes")
        
        def get_ideal_state(used_index: List, available_index: List) -> int:
            us_idx = torch.tensor(used_index)
            av_idx = torch.tensor(available_index).reshape(-1, 1)
            prod_distance = torch.abs(av_idx - us_idx).prod(1)
            return prod_distance.argmax().item()


        target_states = torch.zeros(self.num_classes, 2**self.num_qubits)
        available_index = list(range(2**self.num_qubits))
        used_index = []
        for idx in range(self.num_classes):
            if idx == 0:
                state_idx = 0
            else:
                state_idx = available_index[get_ideal_state(used_index, available_index)]

            target_states[idx, state_idx] = 1
            used_index.append(state_idx)
            available_index.remove(state_idx)

        return target_states
        


    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        
        def loss(output: torch.Tensor) -> torch.Tensor:
            """Computes the fidelity from the circuit output

            Args:
                output (torch.tensor): output of the circuit

            Returns:
                torch.Tensor: fidelities
            """
            return (1.0 - output) ** 2
        
        x = self.pad_layer(x)
        dm_y = self.target_density_matrices[y.int()].squeeze()

        fidelity = vmap(lambda x, dm_y: self.circuit(x, self.q_params, dm_y))(x, dm_y)
        return loss(fidelity)
    

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