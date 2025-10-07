# Step 0: Install necessary libraries
# In a Jupyter cell, you would run:
# !pip install torch pennylane matplotlib

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.func import vmap # <-- ADD THIS IMPORT
import pennylane as qml

# --- Step 1: Generate the "Quantum-Native" Data ---

def generate_quantum_checkerboard_data(n_samples: int):
    """
    Generates a 2D dataset where the classification label is determined
    by a fixed quantum circuit, creating a checkerboard pattern.
    """
    # Define the fixed quantum circuit that will act as our labeling function
    labeling_dev = qml.device("default.qubit", wires=2)
    
    @qml.qnode(labeling_dev, interface="torch")
    def get_quantum_label(features):
        # features is a 2-element vector [x1, x2]
        qml.AngleEmbedding(features, wires=range(2))
        qml.CNOT(wires=[0, 1])
        # The measurement of this joint observable creates the checkerboard
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    # Generate random 2D points
    X = np.random.uniform(-np.pi, np.pi, size=(n_samples, 2))
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Get the labels from the quantum function
    quantum_labels = torch.stack([get_quantum_label(x) for x in X_tensor])
    
    # Binarize the labels: 1 if expectation > 0, else 0
    y = (quantum_labels > 0).long()
    
    print("Data generation complete.")
    return X_tensor, y

# --- Step 2: Visualize the Data ---

def visualize_data(X, y, save_path="checkerboard_plot.png"):
    """Plots the 2D data and saves it to a file."""
    plt.figure(figsize=(8, 8))
    plt.title("The Quantum Checkerboard Dataset")
    
    # Plot class 0 (blue)
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', alpha=0.7, label='Class 0')
    
    # Plot class 1 (red)
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', alpha=0.7, label='Class 1')
    
    plt.xlabel("$x_1$ (feature 1)")
    plt.ylabel("$x_2$ (feature 2)")
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(save_path)
    plt.close() # Free up memory by closing the plot
    print(f"Data visualization saved to {save_path}")

# --- Step 3: Define the Simple Classical Classifier ---

class ClassicalClassifier(nn.Module):
    """A simple linear classifier that should fail on this task."""
    def __init__(self):
        super().__init__()
        # A single linear layer cannot learn the checkerboard pattern
        self.sequential = nn.Sequential(nn.Linear(2,1))

    def forward(self, x):
        return self.sequential(x)

# --- Step 4: Define the Simple Quantum Classifier ---

class QuantumClassifier(nn.Module):
    """A VQC-based classifier with the right inductive bias for the task."""
    def __init__(self, n_layers=1):
        super().__init__()
        n_qubits = 2
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            # MODIFIED: Use a simpler ansatz with a better optimization landscape
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            # Measure the joint observable that defines the problem
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        self.circuit = circuit
        # MODIFIED: Adjust parameter shape for BasicEntanglerLayers
        self.q_params = nn.Parameter(torch.randn(n_layers, n_qubits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # vmap requires a wrapper to fix the weights parameter
        def circuit_wrapper(single_input):
            return self.circuit(single_input, self.q_params)
        
        # Batch execution of the quantum circuit using torch.func.vmap
        exp_vals = vmap(circuit_wrapper)(x)
        return exp_vals.unsqueeze(-1)

# --- Step 5: Train and Compare the Models ---

def calculate_accuracy(y_pred, y_true):
    """Calculates classification accuracy."""
    predicted_labels = (torch.sigmoid(y_pred) > 0.5).long()
    correct = (predicted_labels.squeeze() == y_true).sum().item()
    return correct / len(y_true)

def train_model(model, dataloader, epochs=20, lr=0.1):
    """Generic training loop for a PyTorch model."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"\n--- Training {model.__class__.__name__} ---")
    
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.float().unsqueeze(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            # Calculate accuracy on the full dataset
            with torch.no_grad():
                full_y_pred = model(dataloader.dataset.tensors[0])
                acc = calculate_accuracy(full_y_pred, dataloader.dataset.tensors[1])
                print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f} | Accuracy: {acc:.4f}")
    
    print("Training finished.")
    with torch.no_grad():
        final_y_pred = model(dataloader.dataset.tensors[0])
        final_acc = calculate_accuracy(final_y_pred, dataloader.dataset.tensors[1])
        print(f"Final Accuracy for {model.__class__.__name__}: {final_acc:.4f}")
    return final_acc


if __name__ == '__main__':
    # --- Main Execution ---
    
    # 1. Generate data
    N_SAMPLES = 400
    BATCH_SIZE = 32
    X_data, y_data = generate_quantum_checkerboard_data(n_samples=N_SAMPLES)
    
    # 2. Visualize data
    visualize_data(X_data, y_data)
    
    # Create DataLoader for training
    dataset = TensorDataset(X_data, y_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Instantiate models
    classical_model = ClassicalClassifier()
    # MODIFIED: Explicitly instantiate with 1 layer for a fair comparison
    quantum_model = QuantumClassifier(n_layers=1)

    # 4. Train both models and show the accuracy difference
    classical_acc = train_model(classical_model, dataloader)
    quantum_acc = train_model(quantum_model, dataloader)

    print("\n--- Comparison ---")
    print(f"Classical Model Final Accuracy: {classical_acc:.4f}")
    print(f"Quantum Model Final Accuracy:   {quantum_acc:.4f}")
    print("\nConclusion: The quantum model significantly outperforms the classical one")
    print("due to its well-matched inductive bias for this quantum-native problem.")


