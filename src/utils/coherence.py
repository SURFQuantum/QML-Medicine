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

class CoherenceCalculator:
    def __init__(self, config: dict, backend):
        self.n_quantum_layers = config['n_quantum_layers']
        self.n_qubits = config['n_qubits']
        self.num_classes = config['num_classes']

        props = backend.properties()
        dev_time = [q[1].value for q in props.qubits]  # Decoherence time of the backend used
        self.min_T2 = min(dev_time)
        print(f"Minimum qubit coherence time of chosen backend: {self.min_T2*1e6:.2f} \u03BCs")

        # Gate times in seconds
        self.gate_times = {
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

    def circuit_time(self, qnode, inputs, q_params):
        specs_info = qml.specs(qnode)(inputs, q_params)
        resources = specs_info["resources"]

        total_time = 0.0
        for gate, count in resources.gate_types.items():
            if gate in self.gate_times:
                total_time += count * self.gate_times[gate]

        return total_time


    def forward(self, qnode, inputs, q_params):        
        total_time = self.circuit_time(qnode, inputs, q_params)
        if total_time < self.min_T2:
            print(f"{self.n_quantum_layers} layer(s) can be run on fake backend, cause it doesn't takes too long")
        else:
            print(f"{self.n_quantum_layers} layer(s) cannot be run on fake backend, cause it takes too long")
        return total_time
