import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

from qcm.utils.algebra import hermitianize_image_matrix


# Reuploading QNode factory
def make_reupload_qnode(n_qubits, H, n_reupload=3, time=1.0, n_steps=1, device_name="default.qubit"):
    """
    Build a QNode that repeats: [encode with H] -> [trainable single-qubit RY layer + entangler]
    n_reupload times. Returns a QNode that accepts:
      - sample_idx: integer index into H_list if provided
      - params: flat vector of length (n_reupload * n_qubits)
      - H_list (optional): list of qml.Hamiltonian objects to encode per sample
    Output: expectation value <Z_0> in [-1,1] for classification/regression.
    """
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def qnode(sample_idx, params, H_list=None):
        # prepare equal superposition
        for w in range(n_qubits):
            qml.Hadamard(wires=w)

        p = params.reshape((n_reupload, n_qubits))
        for block in range(n_reupload):
            # choose Hamiltonian (per-sample if H_list provided)
            H_enc = H_list[sample_idx] if (H_list is not None) else H
            qml.templates.ApproxTimeEvolution(H_enc, time, n_steps)

            # trainable single-qubit rotations (one param per qubit)
            for q in range(n_qubits):
                qml.RY(p[block, q], wires=q)

            # simple ring entangler
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])

        return qml.expval(qml.PauliZ(wires=0))

    return qnode

if __name__ == "__main__":
    rng = np.random.RandomState(1)
    N = 32
    n_qubits = int(np.log2(N))

    A = hermitianize_image_matrix(rng.randn(N, N))
    B = hermitianize_image_matrix(rng.randn(N, N) + 0.8)

    H_A = qml.pauli_decompose(A)
    H_B = qml.pauli_decompose(B)

    n_per_class = 6
    H_list = [H_A if i % 2 == 0 else H_B for i in range(2 * n_per_class)]
    labels = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(2 * n_per_class)])

    n_reupload = 3
    qnode = make_reupload_qnode(n_qubits, H_A, n_reupload=n_reupload, time=1.0, n_steps=1)

    n_params = n_reupload * n_qubits
    params = pnp.array(0.05 * rng.randn(n_params), requires_grad=True)

    def cost(p):
        preds = pnp.array([qnode(i, p, H_list=H_list) for i in range(len(H_list))])
        return pnp.mean((preds - labels) ** 2)

    opt = qml.GradientDescentOptimizer(stepsize=0.2)
    print("Initial cost:", float(cost(params)))
    for epoch in range(30):
        params = opt.step(cost, params)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}, cost = {float(cost(params)):.4f}")

    preds = np.array([qnode(i, params, H_list=H_list) for i in range(len(H_list))])
    print("\nFinal predictions (first 10):")
    for i in range(min(10, len(preds))):
        # cast label to int for integer formatting, or use float format
        print(f" label={int(labels[i]):+d}, pred={preds[i]:+.3f}")
