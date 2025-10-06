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

from qcm.data.datasets import get_dataloaders
from qcm.model.reupload import HybridReuploadClassifier # Updated model import
from qcm.utils.functions import visualize_latents, log_losses_to_csv
from qcm.utils.algebra import compute_density_matrix

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_prediction_metric(model, x):
    with torch.no_grad():        
        nbatch = x.shape[0]
        nstate = model.head.num_classes
        target_labels = model.head.target_labels.reshape(1, -1)
        x = torch.repeat_interleave(x, nstate, dim=0)
        y = torch.repeat_interleave(target_labels, nbatch, dim=0).reshape(-1 ,1)
        output = model(x, y).reshape(-1, nstate)
        val_loss, pred = output.min(dim=1) 
        return val_loss, pred
    
def load_config(path: str) -> dict:
    logger.info(f"Loading config from: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train_epoch(model: nn.Module, 
                dataloader: DataLoader, 
                optimizer: torch.optim.Optimizer, 
                scheduler, 
                epoch: int, 
                config: dict, 
                writer: SummaryWriter):
    
    metric_train = BinaryF1Score().to(device)
    metric_name = "F1 Score"
    logger.info(f"Starting training epoch {epoch}")
    model.train()
    losses = []
    throughput = Throughput()
    start = time.monotonic()

    logger.info(f"Epoch {epoch} START")
    for i, (x, y) in enumerate(dataloader):

        x, y = x.to(device), y.long().to(device)
        _, pred = compute_prediction_metric(model, x) 
        metric_train.update(pred, y)
        loss = model(x, y).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        
        if i % 100 == 0 and i > 0:
            throughput.update(i * config['training']['batch_size'], time.monotonic() - start)
            writer.add_scalar('Train/Loss', np.mean(losses[-100:]), epoch * len(dataloader) + i)
            writer.add_scalar('Train/Throughput', throughput.compute(), epoch * len(dataloader) + i)
            logger.info(f"Epoch {epoch}, Step {i}: Loss={loss.item():.4f}")

    metric_train_val = metric_train.compute()
    logger.info(f"Epoch {epoch} END: Loss={loss.item():.4f}, {metric_name}={metric_train_val.item():.4f}")
    scheduler.step()
    logger.info(f"Finished training epoch {epoch}. Avg Loss: {np.mean(losses):.4f}, Accuracy: {metric_train_val.item():.4f}")
    return np.mean(losses)


def validate(model: nn.Module, 
             dataloader: DataLoader, 
             epoch: int, 
             config: dict, 
             writer: SummaryWriter):
    logger.info(f"Starting validation for epoch {epoch}")
    model.eval()
    val_losses = []
    
    # Select metric based on dataset type
    metric = BinaryF1Score().to(device)
    metric_name = "F1 Score"

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            val_loss, pred = compute_prediction_metric(model, x) 
            metric.update(pred, y)
            val_losses.append(val_loss.mean().item())

    metric_val = metric.compute()
    avg_val_loss = np.mean(val_losses)
    
    writer.add_scalar('Val/Loss', avg_val_loss, epoch)
    writer.add_scalar(f'Val/{metric_name}', metric_val.item(), epoch)
    logger.info(f"Validation Epoch {epoch}: Loss={avg_val_loss:.4f}, {metric_name}={metric_val.item():.4f}")
    
    return avg_val_loss


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['classical', 'quantum'], default='classical')
    args = parser.parse_args()
    
    config = load_config(args.config)
    dataset_type = config.get('dataset_type', 'pcam')

    # Define run name and log file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{dataset_type}_{args.mode}_{timestamp}"
    log_filepath = Path("./logs/training_log.csv")
    log_filepath.parent.mkdir(exist_ok=True)

    use_quantum = args.mode == 'quantum'
    logger.info(f"Initializing model for dataset '{dataset_type}' in {'quantum' if use_quantum else 'classical'} mode")
    model = HybridReuploadClassifier(config=config, use_quantum=use_quantum)
    model.to(device)
    
    logger.info("Loading data...")
    train_loader, val_loader = get_dataloaders(config)

    # Setup optimizer, scheduler, and loss function
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])

    logger.info(f"Using Fidelity loss for {dataset_type}.")

    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    train_epoch_losses = []
    val_epoch_losses = []

    for epoch in range(config['training']['epochs']):
        avg_train_loss = train_epoch(model, train_loader, optimizer, scheduler, epoch, config, writer)
        avg_val_loss = validate(model, val_loader, epoch, config, writer)
        
        train_epoch_losses.append(avg_train_loss)
        val_epoch_losses.append(avg_val_loss)

    log_losses_to_csv(log_filepath, run_name, train_epoch_losses, val_epoch_losses)

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