from qcm.utils.image_grid import grid_shape_for_qubits
import pytest   

def test_grid_shape_for_qubits():
    """Test the grid_shape_for_qubits function."""
    assert grid_shape_for_qubits(1) == (1, 1)
    assert grid_shape_for_qubits(4) == (2, 2)
    assert grid_shape_for_qubits(6) == (2, 3)
    assert grid_shape_for_qubits(12) == (3, 4)
    assert grid_shape_for_qubits(13) == (1, 13)  # prime number
    with pytest.raises(ValueError):
        grid_shape_for_qubits(0)
    with pytest.raises(ValueError):
        grid_shape_for_qubits(-5)