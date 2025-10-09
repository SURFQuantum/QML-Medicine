import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import Throughput
import time
import logging
import numpy as np

class SolverQNN:

    def __init__(self, 
                 model: nn.Module, 
                 dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler,
                 config: dict,
                 writer: SummaryWriter,
                 device: torch.device,
                 logger: logging.Logger
                 ):
        
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.writer = writer
        self.device = device
        self.logger = logger

    
    def train_epoch(self, epoch: int):
        """_summary_

        Args:
            epoch (int): _description_

        Returns:
            _type_: _description_
        """
        
        logger.debug(f"Starting training epoch {epoch}")
        self.train()
        losses = []
        throughput = Throughput()
        start = time.monotonic()

        for i, (x, y) in enumerate(dataloader):
            # Adapt label type for loss function
            if config['dataset_type'] == 'pcam':
                x, y = x.to(device), y.float().to(device)
                output = self(x).squeeze(1)
                loss = criterion(output, y.squeeze())
            else:  # tcga
                x, y = x.to(device), y.long().to(device)
                output = self(x)
                loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if hasattr(self.head, 'q_params'):
                q_params_tensor = self.head.q_params.data.cpu()
                current_angles = q_params_tensor.numpy()

                # Log individual angle values
                # Flatten the tensor to iterate over all angles regardless of shape (N_layers, N_qubits, 3)
                for j, angle in enumerate(current_angles.flatten()):
                    writer.add_scalar(
                        f'Train/Q_Angle_{j}',
                        angle,
                        epoch * len(dataloader) + i
                    )

                # Log the magnitude (L2-norm) of the parameter vector
                angle_magnitude = torch.linalg.norm(q_params_tensor)
                writer.add_scalar(
                    'Train/Q_Angle_Magnitude',
                    angle_magnitude.item(),
                    epoch * len(dataloader) + i
                )

            if i % 100 == 0 and i > 0:
                throughput.update(i * config['training']['batch_size'], time.monotonic() - start)
                writer.add_scalar('Train/Loss', np.mean(losses[-100:]), epoch * len(dataloader) + i)
                writer.add_scalar('Train/Throughput', throughput.compute(), epoch * len(dataloader) + i)
                logger.debug(
                    f"Epoch {epoch}, Step {i}: Loss={loss.item():.4f}, Throughput={throughput.compute():.2f} items/sec")

        scheduler.step()
        
        logger.debug(f"Finished training epoch {epoch}. Avg Loss: {np.mean(losses):.4f}")
        return np.mean(losses)
    
    def validate(self,
                 dataloader: DataLoader, 
                 criterion: nn.Module, 
                 epoch: int, 
                 config: dict,
                 writer: SummaryWriter,
                 device: torch.device,
                 logger: logging.logger):
        """_summary_

        Args:
            model (nn.Module): _description_
            dataloader (DataLoader): _description_
            criterion (nn.Module): _description_
            epoch (int): _description_
            config (dict): _description_
            writer (SummaryWriter): _description_
            device (torch.device): _description_
            logger (logging.logger): _description_

        Returns:
            _type_: _description_
        """
        
        logger.debug(f"Starting validation for epoch {epoch}")
        self.eval()
        val_losses = []

        # Select metric based on dataset type
        if config['dataset_type'] == 'pcam':
            metric = BinaryF1Score().to(device)
            metric_name = "F1 Score"
        else:  # tcga
            num_classes = config['model']['num_classes']
            metric = MulticlassAccuracy(num_classes=num_classes).to(device)
            metric_name = "Accuracy"

        with torch.no_grad():
            for x, y in dataloader:
                if config['dataset_type'] == 'pcam':
                    x, y = x.to(device), y.squeeze().float().to(device)
                    output = self(x).squeeze(1)
                    pred = torch.sigmoid(output) > 0.5
                    val_loss = criterion(output, y)
                else:  # tcga
                    x, y = x.to(device), y.long().to(device)
                    output = self(x)
                    pred = torch.argmax(output, dim=1)
                    val_loss = criterion(output, y)

                metric.update(pred, y)
                val_losses.append(val_loss.item())

        metric_val = metric.compute()
        avg_val_loss = np.mean(val_losses)

        writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        writer.add_scalar(f'Val/{metric_name}', metric_val.item(), epoch)
        logger.info(f"Validation Epoch {epoch}: Loss={avg_val_loss:.4f}, {metric_name}={metric_val.item():.4f}")

        return avg_val_loss