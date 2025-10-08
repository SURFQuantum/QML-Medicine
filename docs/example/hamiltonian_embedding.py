import numpy as np
import pennylane as qml

from qcm.utils.algebra import hermitianize_image_matrix


def make_embedding_qnode(n_qubits, H, time=2.4, n_steps=2):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def embedding():
        # example preparation: put all qubits into equal superposition
        for w in range(n_qubits):
            qml.Hadamard(wires=w)

        # use PennyLane's approximate time evolution (Trotter/Suzuki)
        qml.templates.ApproxTimeEvolution(H, time, n_steps)

        # measure final statevector for inspection
        return qml.state()
    return embedding

# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    # create a small 16x16 real image matrix for testing (replace with actual image)
    rng = np.random.RandomState(seed=1)
    M = rng.randn(16, 16)
    H_mat = hermitianize_image_matrix(M)

    # decompose to qml.Hamiltonian (prune tiny coefficients)
    # H = pauli_decomposition(H_mat, tol=.5)
    H = qml.pauli_decompose(H_mat)
    print("Built Hamiltonian with", len(H.coeffs), "nontrivial Pauli terms")

    # make and run qnode
    qnode = make_embedding_qnode(n_qubits=4, H=H, time=1, n_steps=2)
    state = qnode()
    print(qml.draw(qnode)())
    print("Statevector length:", len(state))
