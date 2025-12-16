# train.py

import time
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryF1Score, MulticlassAccuracy
from torcheval.metrics import Throughput
from argparse import ArgumentParser
import logging
from datetime import datetime

import pennylane as qml
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from utils.coherence import CoherenceCalculator
from data.datasets import get_dataloaders
from model.models import HybridClassifier # Updated model import
from utils.functions import visualize_latents, log_losses_to_csv

#logging.getLogger("qiskit").setLevel(logging.ERROR)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("qiskit").setLevel(logging.WARNING)
logging.getLogger("qiskit_aer").setLevel(logging.WARNING)
logging.getLogger("pennylane").setLevel(logging.WARNING)

def load_config(path: str) -> dict:
    logger.info(f"Loading config from: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def qml_backend(n_qubits: int, shots: int = 1024):
    backend = FakeManilaV2()
    logger.info(f"Using backend: {backend}")

    noise_model = NoiseModel.from_backend(backend)  

    sim_backend = AerSimulator(
        noise_model=noise_model,
        basis_gates=noise_model.basis_gates
    )

    qml_dev = qml.device(
        "qiskit.aer",
        wires=n_qubits,
        backend=sim_backend,
        shots=shots
    )

    return backend, qml_dev


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, scheduler, epoch: int, config: dict, writer: SummaryWriter):
    logger.info(f"Starting training epoch {epoch}")
    model.train()
    losses = []
    throughput = Throughput()
    start = time.monotonic()

    for i, (x, y) in enumerate(dataloader):
        # Adapt label type for loss function
        if config['dataset_type'] == 'pcam':
            x, y = x.to(device), y.float().to(device)
            output = model(x).squeeze(1)
            loss = criterion(output, y.squeeze())
        else: # tcga
            x, y = x.to(device), y.long().to(device)
            output = model(x)
            loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i % 100 == 0 and i > 0:
            throughput.update(i * config['training']['batch_size'], time.monotonic() - start)
            writer.add_scalar('Train/Loss', np.mean(losses[-100:]), epoch * len(dataloader) + i)
            writer.add_scalar('Train/Throughput', throughput.compute(), epoch * len(dataloader) + i)
            logger.info(f"Epoch {epoch}, Step {i}: Loss={loss.item():.4f}, Throughput={throughput.compute():.2f} items/sec")

    scheduler.step()
    logger.info(f"Finished training epoch {epoch}. Avg Loss: {np.mean(losses):.4f}")
    return np.mean(losses)


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, epoch: int, config: dict, writer: SummaryWriter):
    logger.info(f"Starting validation for epoch {epoch}")
    model.eval()
    val_losses = []
    
    # Select metric based on dataset type
    if config['dataset_type'] == 'pcam':
        metric = BinaryF1Score().to(device)
        metric_name = "F1 Score"
    else: # tcga
        num_classes = config['model']['num_classes']
        metric = MulticlassAccuracy(num_classes=num_classes).to(device)
        metric_name = "Accuracy"

    with torch.no_grad():
        for x, y in dataloader:
            if config['dataset_type'] == 'pcam':
                x, y = x.to(device), y.squeeze().float().to(device)
                output = model(x).squeeze(1)
                pred = torch.sigmoid(output) > 0.5
                val_loss = criterion(output, y)
            else: # tcga
                x, y = x.to(device), y.long().to(device)
                output = model(x)
                pred = torch.argmax(output, dim=1)
                val_loss = criterion(output, y)
            metric.update(pred, y)
            val_losses.append(val_loss.item())

    metric_val = metric.compute()
    avg_val_loss = np.mean(val_losses)
    
    writer.add_scalar('Val/Loss', avg_val_loss, epoch)
    writer.add_scalar(f'Val/{metric_name}', metric_val.item(), epoch)
    logger.info(f"Validation Epoch {epoch}: Loss={avg_val_loss:.4f}, {metric_name}={metric_val.item():.4f}")
    
    return avg_val_loss, metric_val

def coherence_check(config, qml_dev, backend, model):
    if not hasattr(model, "head"):
        logger.warning("Model has no quantum head.")
        return

    q_head = model.head
    if not hasattr(q_head, "circuit"):
        logger.warning("Quantum head has no circuit.")
        return

    qnode = q_head.circuit
    q_params = q_head.q_params

    n_qubits = config["model"]["n_qubits"]
    head_type = config["model"].get("quantum_head_type", "amplitude")

    if head_type == "angle":
        dummy_inputs = torch.zeros(n_qubits)
    else:  # amplitude
        dummy_inputs = torch.zeros(2 ** n_qubits)

    logger.info("Running quantum coherence check...")

    calc = CoherenceCalculator(config=config["model"],backend=backend)
    total_gate_time = calc.forward(qnode, dummy_inputs, q_params)

    logger.info(f"Circuit gate time = {total_gate_time * 1e6:.2f} \u03BCs") #\u03BC = \mu

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['classical', 'quantum'], default='classical')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config = load_config(args.config)
    dataset_type = config.get('dataset_type', 'pcam')

    final_epoch = config['training']['epochs']
    log_filepath = Path(f"./logs/training_log_{args.mode}_mode_{final_epoch}_epochs_{timestamp}.csv")
    log_filepath.parent.mkdir(exist_ok=True)

    # Define run name and log file path
    run_name = f"{dataset_type}_{args.mode}_{final_epoch}_epochs_{timestamp}"

    qml_dev = None
    backend = None

    use_quantum = args.mode == 'quantum'
    logger.info(f"Initializing model for dataset '{dataset_type}' in {'quantum' if use_quantum else 'classical'} mode")
    
    if use_quantum:
        n_qubits = config["model"].get("n_qubits", 4)
        shots = config.get("quantum", {}).get("shots", 1024)

        backend, qml_dev = qml_backend(n_qubits=n_qubits, shots=shots)
        
    model = HybridClassifier(config=config, device=qml_dev, use_quantum=use_quantum)
    model.to(device)

    if use_quantum:
        coherence_check(config, qml_dev, backend, model)

    logger.info("Loading data...")
    train_loader, val_loader = get_dataloaders(config)

    # Setup optimizer, scheduler, and loss function
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])

    if dataset_type == 'pcam':
        criterion = nn.BCEWithLogitsLoss()
        logger.info("Using Binary Cross-Entropy loss for PCAM.")
    else: # tcga
        criterion = nn.CrossEntropyLoss()
        logger.info("Using Cross-Entropy loss for TCGA.")

    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    train_epoch_losses = []
    val_epoch_losses = []
    accuracies = []

    for epoch in range(config['training']['epochs']):
        avg_train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, config, writer)
        avg_val_loss, accuracy = validate(model, val_loader, criterion, epoch, config, writer)
        
        train_epoch_losses.append(avg_train_loss)
        val_epoch_losses.append(avg_val_loss)
        accuracies.append(float(accuracy))

    log_losses_to_csv(log_filepath, run_name, train_epoch_losses, val_epoch_losses, accuracies)

    # Save model and visualize latents
    save_dir = Path("./models")
    save_dir.mkdir(exist_ok=True)
    model_save_path = save_dir / f"model_backbone_{run_name}.pt"
    torch.save(model.backbone.state_dict(), model_save_path)
    logger.info(f"Model backbone saved to {model_save_path}")

    # Note: visualize_latents might need adjustment for TCGA if it assumes image inputs
    # For now, we assume it works on the latent space directly, which should be fine.
    #if config['dataset_type'] == 'pcam':
    visualize_latents(model, val_loader, f"./models/latent_tsne_{run_name}.png", device=device)
    logger.info("Latent visualization saved.")

    writer.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()