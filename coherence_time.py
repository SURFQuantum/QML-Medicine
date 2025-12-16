import pennylane as qml
import pickle 
import torch.nn as nn
import torch 
import numpy as np
from torch.func import vmap
import math
from pennylane import specs
import matplotlib.pyplot as plt

from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

with open("src/data/tcga/testdata.pkl", "rb") as f: 
    data = pickle.load(f) 
inputs = torch.tensor(data["X"], dtype=torch.float32)

fake_backend = FakeManilaV2() # Fake backend
print(type(fake_backend))
noise_model = NoiseModel.from_backend(fake_backend) # noise model

tot_layers = 10
n_layers_list = np.arange(1,tot_layers+1)
print('Amount of layers printed')
n_qubits = 8 # Amount of qubits

props = fake_backend.properties()
T2_list = [q[1].value for q in props.qubits]  # Decoherence time
min_T2 = min(T2_list)
print(f"Minimum qubit coherence time of chosen backend: {min_T2*1e6:.2f} \u03BCs")

# Function for estimating coherence time
def circuit_time(qnode, depth):
    # Gate times in seconds
    gate_times = {
        "Rot": 50e-9,
        "RX": 50e-9,
        "RY": 50e-9,
        "RZ": 50e-9,
        "CNOT": 300e-9,
        "CZ": 300e-9,
        "Hadamard": 50e-9,
        "PauliX": 50e-9,
        "PauliY": 50e-9,
        "PauliZ": 50e-9,
    }

    tape = qnode._tape.expand(depth=depth)

    # Calculating the full time of the circuit
    total_time = 0
    for op in tape.operations:
        if op.name in gate_times:
            total_time += gate_times[op.name]

    print(f"Total circuit time: {total_time*1e6:.2f} \u03BCs")
    return total_time
    
class FeatureCompressor(nn.Module):
        def __init__(self, input_dim=len(inputs[0]), n_qubits=n_qubits):
            super().__init__()
            self.compress = nn.Linear(input_dim, n_qubits)
        def forward(self, x):
            return self.compress(x)

coherence_times_list = []
for n_layers in n_layers_list:
    # Simulated backend
    sim_backend = AerSimulator(
        noise_model=noise_model,
        basis_gates=noise_model.basis_gates
    )

    # Device
    dev = qml.device(
        "qiskit.aer",
        wires=n_qubits,
        backend=sim_backend,
        shots=1024
    )

    q_params = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
    entangler = qml.templates.StronglyEntanglingLayers

    compressor = FeatureCompressor(input_dim=inputs.shape[1], n_qubits=n_qubits)
    inputs_reduced = compressor(inputs).float()

    quantum_features_list = []
    @qml.qnode(dev, interface="torch", diff_method="finite-diff")
    # Random circuit
    def circuit(inputs, q_params_):
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
        entangler(q_params_, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    for x_sample in inputs_reduced:
        out = circuit(x_sample, q_params)
        quantum_features_list.append(torch.tensor(out, dtype=torch.float32))

    quantum_features = torch.stack(quantum_features_list, dim=0)

    num_classes = 33
    classifier = nn.Linear(n_qubits, num_classes)
    logits = classifier(quantum_features.float())

    specs_func = qml.specs(circuit)
    specs_info = specs_func(inputs_reduced[0], q_params)
    depth = specs_info["resources"].depth

    # Coherence estimate
    total_time = circuit_time(circuit, depth)
    if total_time < min_T2:
        print(f"{n_layers} layer(s) can be run on fake backend, cause it doesn't takes too long")
    else:
        print(f"{n_layers} layer(s) cannot be run on fake backend, cause it takes too long")

    coherence_times_list.append(total_time)

for i in range(len(coherence_times_list)):
    coherence_times_list[i] = coherence_times_list[i]*1e6

plt.plot(n_layers_list,coherence_times_list)
plt.xlabel('Layers')
plt.ylabel('Coherence time (\u03BCs)')
plt.title('Coherence time over layers')
plt.grid()
plt.savefig(f'plots/coherence_times_over_layers.png',dpi=120)
plt.show()