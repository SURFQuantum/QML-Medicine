
import torch
import torch.nn as nn
import pennylane as qml

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