import torch
import torch.nn as nn
import pennylane as qml
from typing import Tuple
import math

from ..utils.image_grid import grid_shape_for_qubits

class QuantumHeadHamiltonianPaper(nn.Module):
    """Hamiltonian embedding for a quantum node as close as possible to the paper: https://arxiv.org/pdf/2407.14055"""
    def __init__(self, n_qubits: int, num_classes: int,
                 n_layers: int = 1, entangling_layer: str = "strong",
                 include_xx: bool = True, n_trotter_steps: int = 1, time: float = 1.0):
        super().__init__()
        self.n_qubits = n_qubits
        self.time = float(time)
        self.n_trotter_steps = int(n_trotter_steps)
        self.include_xx = bool(include_xx)

        # --- FIX: qubit grid (area = n_qubits) ---
        Hq, Wq = grid_shape_for_qubits(n_qubits)
        self.Hq, self.Wq = Hq, Wq
        self.pairs = []
        for r in range(Hq):
            for c in range(Wq):
                idx = r * Wq + c
                if c + 1 < Wq:
                    self.pairs.append((idx, r * Wq + (c + 1)))
                if r + 1 < Hq:
                    self.pairs.append((idx, (r + 1) * Wq + c))
        self.n_pairs = len(self.pairs)

        self.required_latent_dim = 2 ** n_qubits
        out_dim = self.n_qubits + self.n_pairs + (self.n_pairs if self.include_xx else 0)
        self.coeff_layer = nn.Linear(self.required_latent_dim, out_dim)

        if entangling_layer == "strong":
            self.q_params = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
            self.entangler = qml.templates.StronglyEntanglingLayers
        elif entangling_layer == "basic":
            self.q_params = nn.Parameter(torch.randn(n_layers, n_qubits))
            self.entangler = qml.templates.BasicEntanglerLayers
        else:
            raise ValueError(f"Unknown entangling_layer: {entangling_layer}")

        dev = qml.device("default.qubit", wires=list(range(n_qubits)))

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(flat_coeffs, q_params):
            alpha = [flat_coeffs[i] for i in range(self.n_qubits)]
            offset = self.n_qubits
            beta = [flat_coeffs[offset + k] for k in range(self.n_pairs)]
            offset += self.n_pairs
            gamma = [flat_coeffs[offset + k] for k in range(self.n_pairs)] if self.include_xx else []

            for w in range(self.n_qubits):
                qml.Hadamard(wires=w)

            dt = self.time / max(1, self.n_trotter_steps)
            for _ in range(self.n_trotter_steps):
                for i in range(self.n_qubits):
                    qml.RZ(2.0 * alpha[i] * dt, wires=i)
                for k, (i, j) in enumerate(self.pairs):
                    qml.CNOT(wires=[i, j])
                    qml.RZ(2.0 * beta[k] * dt, wires=j)
                    qml.CNOT(wires=[i, j])
                if self.include_xx:
                    for k, (i, j) in enumerate(self.pairs):
                        qml.Hadamard(wires=i); qml.Hadamard(wires=j)
                        qml.CNOT(wires=[i, j])
                        qml.RZ(2.0 * gamma[k] * dt, wires=j)
                        qml.CNOT(wires=[i, j])
                        qml.Hadamard(wires=i); qml.Hadamard(wires=j)

            self.entangler(q_params, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit
        self.classifier = nn.Linear(n_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("Input x must be 2D (batch_size, latent_dim)")
        batch_size, input_dim = x.shape
        if input_dim != self.required_latent_dim:
            raise ValueError("Input dim mismatch for HamiltonianPaper.")

        coeffs_all = self.coeff_layer(x)
        outputs = []
        for b in range(batch_size):
            flat = coeffs_all[b]
            out = self.circuit(flat, self.q_params)
            outputs.append(torch.stack(out))
        quantum_features = torch.stack(outputs, dim=0).float()
        return self.classifier(quantum_features)