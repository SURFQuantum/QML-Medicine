import numpy as np
import pennylane as qml

def hermitianize_image_matrix(image_matrix: np.ndarray) -> np.ndarray:
    """Make an arbitrary square matrix Hermitian (H = (M + M^†)/2)."""
    return 0.5 * (image_matrix + image_matrix.conj().T)