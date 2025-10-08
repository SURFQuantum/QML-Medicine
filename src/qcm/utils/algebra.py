import numpy as np
import torch
import pennylane as qml

def hermitianize_image_matrix(image_matrix: np.ndarray) -> np.ndarray:
    """Make an arbitrary square matrix Hermitian (H = (M + M^†)/2)."""
    return 0.5 * (image_matrix + image_matrix.conj().T)

def compute_density_matrix(state):
    """Calculates the density matrix representation of a state.

    Args:
        state (array[complex]): array representing a quantum state vector

    Returns:
        dm: (array[complex]): array representing the density matrix
    """
    return state * torch.conj(state).T

# def pad_data(data):
#     if data.shape[1]%3 != 0:
#         nadd = 3-data.shape[1]%3
#         data = torch.hstack((data, torch.zeros((data.shape[0], nadd), 
#                                          requires_grad=False)))
#     return data