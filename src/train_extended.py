import time
from pathlib import Path

import torch
import torch.nn as nn
import logging
from datetime import datetime
from typing import Dict, Any, List

# Set device and logging (Keep these consistent)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

from .train import load_config, get_dataloaders, train_epoch, validate, log_losses_to_csv
from model.models import HybridClassifier

def run_qcnn_training_job(config: dict, use_quantum: bool, run_name: str) -> Dict[str, Any]:
    """
    Executes the full QCNN training and validation loop using imported functions.

    CORRECTION: Captures and passes the full list of training losses to log_losses_to_csv.
    """

    dataset_type = config.get('dataset_type', 'pcam')

    # 1. Initialization and Setup
    logger.info(f"Initializing model for '{dataset_type}' in {'quantum' if use_quantum else 'classical'} mode")
    model = HybridClassifier(config=config, use_quantum=use_quantum)
    model.to(device)

    logger.info("Loading data...")
    train_loader, val_loader = get_dataloaders(config)

    # Setup optimizer, scheduler, loss function
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])

    if dataset_type == 'pcam':
        criterion = nn.BCEWithLogitsLoss()
    else:  # tcga
        criterion = nn.CrossEntropyLoss()

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    # --- START BENCHMARK TIME CAPTURE ---
    start_time = time.monotonic()

    # ADDED: Initialize list for training losses
    train_epoch_losses: List[float] = []
    val_epoch_losses: List[float] = []

    # 2. Main Training Loop
    for epoch in range(config['training']['epochs']):
        # Execute training and capture loss
        avg_train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, config, writer)
        train_epoch_losses.append(avg_train_loss)  # <--- CAPTURE TRAIN LOSS

        # Execute validation and capture loss
        avg_val_loss = validate(model, val_loader, criterion, epoch, config, writer)
        val_epoch_losses.append(avg_val_loss)

        logger.info(f"Epoch {epoch}: Val Loss={avg_val_loss:.4f}")

    # --- END BENCHMARK TIME CAPTURE ---
    end_time = time.monotonic()

    # 3. Post-Training Cleanup
    log_filepath = Path("./logs/training_log.csv")

    # CORRECTED: Pass the full list of train_epoch_losses instead of [0]
    log_losses_to_csv(log_filepath, run_name, train_epoch_losses, val_epoch_losses)

    # ... (Add model saving logic from original train.py here) ...

    writer.close()
    logger.info("Training complete.")

    # 4. Return Data for Benchmark Framework
    return {
        'cost_history': val_epoch_losses,
        'start_time': start_time,
        'end_time': end_time,
    }