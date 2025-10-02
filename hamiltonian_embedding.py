import numpy as np
import pennylane as qml
from itertools import product
from functools import reduce
from typing import List, Tuple

# Single-qubit Pauli ops (for numpy-level matrix builds)
PAULI_NP = {
    "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1j], [1j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}

# PennyLane operator constructors for wires
PAULI_QML = {
    "I": lambda w: qml.Identity(wires=w),
    "X": lambda w: qml.PauliX(wires=w),
    "Y": lambda w: qml.PauliY(wires=w),
    "Z": lambda w: qml.PauliZ(wires=w),
}

def hermitianize_image_matrix(image_matrix: np.ndarray) -> np.ndarray:
    """Make an arbitrary square matrix Hermitian (H = (M + M^†)/2)."""
    return 0.5 * (image_matrix + image_matrix.conj().T)

def pauli_decomposition(
    h_mat: np.ndarray, tol: float = 1e-10
) -> qml.Hamiltonian:
    """
    Full Pauli-basis decomposition of an N x N Hermitian matrix H (N = 2^n).
    Returns a qml.Hamiltonian with terms pruned by `tol`, ordered by abs(coeff) desc.
    """
    H = np.array(h_mat, dtype=complex)
    N = H.shape[0]
    assert H.shape[0] == H.shape[1], "H must be square"
    n = int(np.round(np.log2(N)))
    assert 2**n == N, "Matrix dimension must be a power of two (N = 2^n)"

    coeffs = []
    ops = []

    for chars in product("IXYZ", repeat=n):
        # skip identity-only string (global phase)
        if all(c == "I" for c in chars):
            continue

        # build numpy Pauli matrix for chars via kron
        P = PAULI_NP[chars[0]]
        for ch in chars[1:]:
            P = np.kron(P, PAULI_NP[ch])

        alpha = np.trace(P.conj().T @ H) / (2**n)
        if abs(alpha) <= tol:
            continue

        # build PennyLane operator: combine single-qubit ops using @ (tensor product)
        qml_ops = []
        for w, ch in enumerate(chars):
            if ch != "I":
                qml_ops.append(PAULI_QML[ch](w))
        if len(qml_ops) == 0:
            # would be identity-only, skipped above
            continue
        elif len(qml_ops) == 1:
            op = qml_ops[0]
        else:
            # reduce with operator '@' to build a multi-qubit operator
            op = reduce(lambda a, b: a @ b, qml_ops)

        coeffs.append(float(np.real_if_close(alpha)))
        ops.append(op)

    # order terms by descending absolute coefficient
    if len(coeffs) > 0:
        order = np.argsort(np.abs(coeffs))[::-1]
        coeffs = [coeffs[i] for i in order]
        ops = [ops[i] for i in order]
        return qml.Hamiltonian(coeffs, ops)
    else:
        # no non-trivial terms, return a trivial Hamiltonian
        return qml.Hamiltonian([0.0], [qml.Identity(wires=0)])


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
