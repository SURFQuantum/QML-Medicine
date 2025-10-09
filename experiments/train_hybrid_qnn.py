# train.py

import time
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryF1Score, MulticlassAccuracy
from torcheval.metrics import Throughput
from argparse import ArgumentParser
import logging
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from qcm.data.datasets import get_dataloaders
from qcm.models.hybrid_qnn import HybridClassifier
from qcm.utils.functions import (visualize_latents, log_losses_to_csv, 
                                 plot_all_runs_losses, load_config)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['classical', 'quantum'], default='quantum')
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_type = config.get('dataset_type', 'pcam')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filepath = Path("./logs/training_log.csv")
    log_filepath.parent.mkdir(exist_ok=True)
    max_num_of_quantum = 5

    use_quantum_mode = args.mode == 'quantum'
    use_quantum = 'quantum'
    all_runs_losses = {}

    for n_quantum_layers in range(max_num_of_quantum + 1):
        if not use_quantum_mode and n_quantum_layers > 0:
            logger.info(f"Skipping run for {n_quantum_layers} Q Layers as mode is 'classical'.")
            continue  # Skip runs with quantum layers if mode is 'classical'

        current_num_quantum_layers = n_quantum_layers
        current_use_quantum = current_num_quantum_layers > 0
        mode_str = 'classical' if current_num_quantum_layers == 0 else 'quantum'
        run_name = f"{dataset_type}_{mode_str}_Q{current_num_quantum_layers}_{config['model']['quantum_head_type']}_{timestamp}"

        logger.info(
            f"\n--- Starting run for **num_quantum_layers = {current_num_quantum_layers}** (Mode: {'Hybrid' if current_use_quantum else 'Classical'}) ---")
        logger.info(
            f"Initializing model for dataset '{dataset_type}' in {'quantum' if current_use_quantum else 'classical'} mode with {current_num_quantum_layers} quantum layers")

        # NOTE: The model must be initialized with use_quantum=True ONLY if n_quantum_layers > 0
        model = HybridClassifier(config=config, use_quantum=current_use_quantum,
                                 num_quantum_layers=current_num_quantum_layers)
        model.to(device)
        logger.debug("Loading data...")
        train_loader, val_loader = get_dataloaders(config)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['lr'], weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)

        if dataset_type == 'pcam':
            criterion = nn.BCEWithLogitsLoss()
            logger.info("Using Binary Cross-Entropy loss for PCAM.")
        else:  # tcga
            criterion = nn.CrossEntropyLoss()
            logger.info("Using Cross-Entropy loss for TCGA.")

        writer = SummaryWriter(log_dir=f"runs/{run_name}")
        train_epoch_losses = []
        val_epoch_losses = []

        for epoch in range(config['training']['epochs']):
            avg_train_loss = model.train_epoch(model, train_loader, criterion, 
                                               optimizer, scheduler, epoch, config, writer,
                                               device, logger)
            avg_val_loss = model.validate(model, val_loader, criterion, 
                                          epoch, config, writer,
                                          device, logger)

            train_epoch_losses.append(avg_train_loss)
            val_epoch_losses.append(avg_val_loss)

        # 5. Log and Save Results for this run
        log_losses_to_csv(log_filepath, run_name, train_epoch_losses, val_epoch_losses)

        # Save model and visualize latents
        save_dir = Path("./models")
        save_dir.mkdir(exist_ok=True)
        model_save_path = save_dir / f"model_backbone_{run_name}.pt"
        # Ensure the model is saved correctly (using state_dict is common)
        try:
            torch.save(model.backbone.state_dict(), model_save_path)
            logger.info(f"Model backbone saved to {model_save_path}")
        except AttributeError:
            # Handle cases where `model.backbone` might not exist or be accessible
            torch.save(model.state_dict(), model_save_path)
            logger.warning(f"Could not save model.backbone. Saved full model state_dict to {model_save_path}")

        # Visualize latents
        visualize_latents(model, val_loader, f"./models/latent_tsne_{run_name}.png", device=device)
        logger.info("Latent visualization saved.")

        writer.close()
        all_runs_losses[current_num_quantum_layers] = (train_epoch_losses, val_epoch_losses)

    logger.info("\n====================================")
    logger.info("All training runs complete.")
    logger.info("====================================")
    logger.info(f"Losses collected for {len(all_runs_losses)} runs: {list(all_runs_losses.keys())} quantum layers.")

    plot_save_path = Path(f"./results/aggregate_loss_plot_{dataset_type}_{timestamp}.png")
    plot_save_path.parent.mkdir(exist_ok=True)
    plot_all_runs_losses(all_runs_losses, dataset_type, timestamp, plot_save_path)



if __name__ == "__main__":
    main()