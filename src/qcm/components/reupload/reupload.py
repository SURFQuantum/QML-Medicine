import torch
import torch.nn as nn
from torch.func import vmap 
import pennylane as qml
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
        def circuit(inputs, q_params_, y):
            """A variational quantum circuit representing the Universal classifier.

            Args:
                params (array[float]): array of parameters
                x (array[float]): single input vector
                y (array[float]): single output state density matrix

            Returns:
                float: fidelity between output state and input
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

    def get_pad_size(self):
        """Get the padding size for the input data."""
        mult = 3 * self.num_qubits
        return 3 - self.num_features % mult        

    def get_target_labels(self):
        if self.num_classes == 2:
            return torch.tensor([0,1]).reshape(-1,1)
        else:
            raise ValueError("Reupload only works with two classes so far")        

    def get_target_states(self):
        if self.num_classes == 2 :
            
            zeros = torch.zeros(2**self.num_qubits, 1)
            zeros[0] = 1

            ones = torch.zeros(2**self.num_qubits, 1)
            ones[-1] = 1

            return torch.stack((zeros, ones))
        else:
            raise ValueError("Reupload only works with two classes so far")


    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        
        def loss(output):
            return (1.0 - output) ** 2
        
        x = self.pad_layer(x)
        dm_y = self.target_density_matrices[y.int()].squeeze()

        fidelity = vmap(lambda x, dm_y: self.circuit(x, self.q_params, dm_y))(x, dm_y)
        return loss(fidelity)
    

    

