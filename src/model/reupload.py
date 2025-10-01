import torch
import torch.nn as nn
from torch.func import vmap 
import pennylane as qml
import pennylane.numpy as np


class QuantumHeadReupload(nn.Module):

    def __init__(self, n_qubits: int, num_classes: int, n_layers: int = 1, entangling_layer: str = 'strong'):
        super().__init__()
        self.n_qubits = n_qubits
        self.num_classes = num_classes
        self.n_layers = n_layers 
        self.required_latent_dim = n_qubits * n_layers * 3

       
        dev = qml.device("default.qubit", wires=n_qubits)
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
            
            n_inp_bloc = -(len(inputs)//-3)
            
            for p in q_params_:
                for i in range(n_inp_bloc):
                    qml.Rot(*inputs[i*3:(i+1)*3], wires=0)
                qml.Rot(*p, wires=0)
            return qml.expval(qml.Hermitian(y, wires=[0]))     
        
        self.circuit = circuit

    def cost(self, params, x, y, state_labels=None):
        """Cost function to be minimized.

        Args:
            params (array[float]): array of parameters
            x (array[float]): 2-d array of input vectors
            y (array[float]): 1-d array of targets
            state_labels (array[float]): array of state representations for labels

        Returns:
            float: loss value to be minimized
        """
        # Compute prediction for each input in data batch
        loss = 0.0
        dm_labels = [self.density_matrix(s) for s in state_labels]
        for i in range(len(x)):
            f = self.qcircuit(params, x[i], dm_labels[y[i]])
            loss = loss + (1 - f) ** 2
        return loss / len(x)

    @staticmethod
    def pad_data(data):
        if data.shape[1]%3 != 0:
            nadd = 3-data.shape[1]%3
            data = np.hstack((data, np.zeros((data.shape[0], nadd), requires_grad=False)))
        return data

    # Define output labels as quantum state vectors
    @staticmethod
    def density_matrix(state):
        """Calculates the density matrix representation of a state.

        Args:
            state (array[complex]): array representing a quantum state vector

        Returns:
            dm: (array[complex]): array representing the density matrix
        """
        return state * np.conj(state).T

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size, input_dim = x.shape
        if input_dim != self.required_latent_dim:
            raise ValueError(
                f"Input dimension ({input_dim}) must match required latent dim "
                f"({self.required_latent_dim} = 3 * {self.n_qubits} * {self.n_layers}) for Reuploading."
            )
        
        return self.circuit(x, self.q_params, y)

        
    
    @staticmethod
    def iterate_minibatches(inputs, targets, batch_size):
        """
        A generator for batches of the input data

        Args:
            inputs (array[float]): input data
            targets (array[float]): targets

        Returns:
            inputs (array[float]): one batch of input data of length `batch_size`
            targets (array[float]): one batch of targets of length `batch_size`
        """
        for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
            idxs = slice(start_idx, start_idx + batch_size)
            yield inputs[idxs], targets[idxs]