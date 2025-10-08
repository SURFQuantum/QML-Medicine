import torch
import torch.nn as nn
from torch.func import vmap 
import pennylane as qml

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