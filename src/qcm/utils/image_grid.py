import torch
import torch.nn as nn
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