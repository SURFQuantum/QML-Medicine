import torch

def compute_density_matrix(state: torch.Tensor) -> torch.Tensor:
    """Calculates the density matrix representation of a state.

    Args:
        state (array[complex]): array representing a quantum state vector

    Returns:
        dm: (array[complex]): array representing the density matrix
    """
    if state.ndim == 1:
        state = state.reshape(-1, 1)
    return state * torch.conj(state).T

def pad_data(data):
    if data.shape[1]%3 != 0:
        nadd = 3-data.shape[1]%3
        data = torch.hstack((data, torch.zeros((data.shape[0], nadd), 
                                         requires_grad=False)))
    return data