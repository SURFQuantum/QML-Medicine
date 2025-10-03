# model/models.py

import torch
import torch.nn as nn
from torch.func import vmap 
import pennylane as qml
from typing import Tuple
import math

# ---------------------------
# Utility
# ---------------------------
def grid_shape_for_amplitude(n_qubits: int) -> Tuple[int, int]:
    """
    Choose a rectangular grid (H, W) such that H*W = 2**n_qubits.
    We split exponents roughly evenly to keep the grid balanced.
    """
    total_exp = n_qubits
    h_exp = total_exp // 2
    w_exp = total_exp - h_exp
    H, W = 2 ** h_exp, 2 ** w_exp
    return H, W

def grid_shape_for_qubits(n_qubits: int) -> Tuple[int, int]:
    """
    Choose a rectangular grid (H, W) such that H*W == n_qubits.
    Prefer balanced shapes; if n_qubits is prime, fallback to (1, n_qubits).
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    # Try to find factors close to sqrt(n_qubits)
    root = int(math.floor(math.sqrt(n_qubits)))
    for h in range(root, 0, -1):
        if n_qubits % h == 0:
            return h, n_qubits // h
    return 1, n_qubits

# =============================================================================
# Backbones (modified to support optional spatial-preserving output)
# =============================================================================
class PCAMBackbone(nn.Module):
    def __init__(self, latent_dim: int = 64, filters: int = 4,
                 preserve_spatial: bool = False, out_grid: Tuple[int, int] = (2, 2)):
        """
        If preserve_spatial is False (default), behavior matches the original:
          - AdaptiveAvgPool2d((2,2)), flatten, then Linear -> latent_dim.
        If preserve_spatial is True, we:
          - AdaptiveAvgPool2d(out_grid), apply a 1x1 conv to produce one scalar per cell,
            and return the flattened grid of length H*W (no final dense mixing).
        This lets the caller control whether the backbone preserves 2D locality.
        """
        super().__init__()
        multiplier = 2
        self.preserve_spatial = bool(preserve_spatial)
        self.out_grid = out_grid

        self.features = nn.Sequential(
            nn.Conv2d(3, filters, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters), nn.ReLU(),
            nn.Conv2d(filters, filters*multiplier, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters*multiplier), nn.ReLU(),
            nn.Conv2d(filters*multiplier, filters*multiplier**2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters*multiplier**2), nn.ReLU()
        )

        if self.preserve_spatial:
            H, W = out_grid
            # Replace the final pooling with one that outputs the desired grid
            self.pool = nn.AdaptiveAvgPool2d((H, W))
            # project channel dimension to 1 per grid cell (a scalar per cell)
            self.cell_proj = nn.Conv2d(filters*multiplier**2, 1, kernel_size=1)
            # No final linear mixing — return flattened spatial map of length H*W
            self.latent = None
        else:
            # Preserve original behavior
            self.pool = nn.AdaptiveAvgPool2d((2, 2))
            self.latent = nn.Sequential(
                nn.Linear(filters*multiplier**2 * 2 * 2, latent_dim),
                nn.Tanh()
            )
            self.cell_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        if self.preserve_spatial:
            x = self.cell_proj(x)         # (B,1,H,W)
            x = x.view(x.size(0), -1)     # (B, H*W)
            return x
        else:
            x = x.view(x.size(0), -1)
            return self.latent(x)

class TCGABackbone(nn.Module):
    def __init__(self, input_dim: int = 768, latent_dim: int = 16,
                 preserve_spatial: bool = False, out_grid: Tuple[int, int] = (4, 4)):
        """
        If preserve_spatial is False (default), matches original projection -> latent_dim.
        If preserve_spatial is True, we project to a vector of length H*W and return it
        (keeping a spatial layout for Hamiltonian construction). Caller must ensure
        latent_dim == H*W when using preserve_spatial=True.
        """
        super().__init__()
        self.preserve_spatial = bool(preserve_spatial)
        self.out_grid = out_grid
        if self.preserve_spatial:
            H, W = out_grid
            flat_dim = H * W
            self.projection = nn.Sequential(
                nn.Linear(input_dim, flat_dim),
                nn.LayerNorm(flat_dim),
                nn.ReLU(),
                nn.Tanh()
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.ReLU(),
                nn.Tanh()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

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
        # raw_features is a LIST of tensors, one for each measurement.
        raw_features = self.circuit(x, self.q_params)
        # Stack the list into a single tensor. Shape becomes (n_qubits, batch_size)
        stacked_features = torch.stack(raw_features, dim=0)
        # Permute to (batch_size, n_qubits)
        quantum_features = stacked_features.permute(1, 0).float()
        return self.classifier(quantum_features)


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


# =============================================================================
# Main Classifier (only applies spatial-preserving backbone when head_type == 'hamiltonian')
# =============================================================================
class HybridClassifier(nn.Module):
    def __init__(self, config: dict, use_quantum: bool = False):
        super().__init__()
        model_cfg = config['model']
        dataset_type = config['dataset_type']

        # 1. Default Backbone selection (will be overridden if use_quantum and head_type requires different latent)
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
                    n_layers=model_cfg['n_quantum_layers'],
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
                    n_layers=model_cfg['n_quantum_layers'],
                    entangling_layer=model_cfg['entangling_layer']
                )


            elif head_type == 'hamiltonian':
                latent_dim = 2 ** n_qubits
                H, W = grid_shape_for_amplitude(n_qubits)
                if dataset_type == 'tcga':
                    self.backbone = TCGABackbone(input_dim=768, latent_dim=latent_dim,
                                                 preserve_spatial=True, out_grid=(H, W))
                else:
                    self.backbone = PCAMBackbone(latent_dim=latent_dim,
                                                 filters=model_cfg['pcam_filters'],
                                                 preserve_spatial=True, out_grid=(H, W))
                variant = model_cfg.get('hamiltonian_variant', 'simple')  # 'simple' or 'paper'
                if variant == 'simple':
                    self.head = QuantumHeadHamiltonianSimple(
                        n_qubits=n_qubits,
                        num_classes=num_classes,
                        n_layers=model_cfg['n_quantum_layers'],
                        entangling_layer=model_cfg['entangling_layer'],
                        include_zz=model_cfg.get('hamiltonian_include_zz', True),
                        time=model_cfg.get('hamiltonian_time', 1.0)
                    )
                elif variant == 'paper':
                    self.head = QuantumHeadHamiltonianPaper(
                        n_qubits=n_qubits,
                        num_classes=num_classes,
                        n_layers=model_cfg['n_quantum_layers'],
                        entangling_layer=model_cfg['entangling_layer'],
                        include_xx=model_cfg.get('hamiltonian_include_xx', True),
                        n_trotter_steps=model_cfg.get('hamiltonian_trotter_steps', 1),
                        time=model_cfg.get('hamiltonian_time', 1.0)
                    )
                else:
                    raise ValueError(f"Unknown hamiltonian_variant: {variant}")
            else:
                raise ValueError(f"Unknown quantum_head_type: {head_type}")
        else:
            latent_dim = model_cfg['latent_dim']
            self.head = ClassicalHead(latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x)
