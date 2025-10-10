import torch
import torch.nn as nn
import pennylane as qml
from typing import Tuple
import math

from ...utils.image_grid import grid_shape_for_qubits

class QuantumHeadHamiltonianSimple(nn.Module):
    """Hamiltonian embedding for a quantum node simplified and lighter to compute."""

    def __init__(self, n_qubits: int, num_classes: int,
                 n_layers: int = 1, entangling_layer: str = "strong",
                 include_zz: bool = True, time: float = 1.0):
        super().__init__()
        self.n_qubits = n_qubits
        self.num_classes = num_classes
        self.time = float(time)
        self.include_zz = bool(include_zz)

        # --- qubit grid (area = n_qubits), not amplitude grid (2**n_qubits) ---
        Hq, Wq = grid_shape_for_qubits(n_qubits)
        self.Hq, self.Wq = Hq, Wq
        self.pairs = []
        for r in range(Hq):
            for c in range(Wq):
                idx = r * Wq + c                # 0 .. n_qubits-1
                if c + 1 < Wq:
                    self.pairs.append((idx, r * Wq + (c + 1)))
                if r + 1 < Hq:
                    self.pairs.append((idx, (r + 1) * Wq + c))
        self.n_pairs = len(self.pairs)

        # Keep input compatibility: backbone still outputs length 2**n_qubits
        self.required_latent_dim = 2 ** n_qubits
        self.coeff_layer = nn.Linear(self.required_latent_dim, self.n_qubits + self.n_pairs)

        if entangling_layer == "strong":
            self.q_params = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
            self.entangler = qml.templates.StronglyEntanglingLayers
        elif entangling_layer == "basic":
            self.q_params = nn.Parameter(torch.randn(n_layers, n_qubits))
            self.entangler = qml.templates.BasicEntanglerLayers
        else:
            raise ValueError(f"Unknown entangling_layer: {entangling_layer}")

        # Make device wires explicit (safe)
        dev = qml.device("default.qubit", wires=list(range(n_qubits)))

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(flat_coeffs, q_params):
            for w in range(self.n_qubits):
                qml.Hadamard(wires=w)

            alpha = [flat_coeffs[i] for i in range(self.n_qubits)]
            beta = [flat_coeffs[self.n_qubits + k] for k in range(self.n_pairs)]

            for i in range(self.n_qubits):
                qml.RZ(2.0 * alpha[i] * self.time, wires=i)

            for k, (i, j) in enumerate(self.pairs):
                qml.CNOT(wires=[i, j])
                qml.RZ(2.0 * beta[k] * self.time, wires=j)
                qml.CNOT(wires=[i, j])

            self.entangler(q_params, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit
        self.classifier = nn.Linear(n_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("Input x must be 2D (batch_size, latent_dim)")
        batch_size, input_dim = x.shape
        if input_dim != self.required_latent_dim:
            raise ValueError("Input dim mismatch for HamiltonianSimple.")

        coeffs_all = self.coeff_layer(x)  # (batch, n_qubits + n_pairs)
        outputs = []
        for b in range(batch_size):
            flat = coeffs_all[b]
            out = self.circuit(flat, self.q_params)
            outputs.append(torch.stack(out))
        quantum_features = torch.stack(outputs, dim=0).float()
        return self.classifier(quantum_features)

    def draw_circuit(self, max_expansion=1):
        """Draw the circuit for visualization purposes."""
        x = torch.rand(self.required_latent_dim)
        param = torch.rand(self.q_params.shape)
        print(qml.draw_mpl(qml.transforms.decompose(self.circuit, 
                                                    max_expansion = max_expansion))(x, param))