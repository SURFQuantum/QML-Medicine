import pytest
import numpy as np 
from qcm.utils.algebra import compute_density_matrix, hermitianize_image_matrix

def test_hermitianize_image_matrix():
    """Test the hermitianize_image_matrix function."""
    matrix = np.array([[1, 2 + 1j], [3 - 1j, 4]])
    hermitian_matrix = hermitianize_image_matrix(matrix)
    assert np.allclose(hermitian_matrix, hermitian_matrix.conj().T), "Matrix is not Hermitian"
    expected = np.array([[1, 2 + 0j], [2 + 0j, 4]])
    assert np.allclose(hermitian_matrix, expected), "Hermitian matrix not as expected"

def test_compute_density_matrix():
    """Test the compute_density_matrix function."""
    state = np.array([[1/np.sqrt(2)], [1/np.sqrt(2)]])
    density_matrix = compute_density_matrix(state)
    expected = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert np.allclose(density_matrix, expected), "Density matrix not as expected"