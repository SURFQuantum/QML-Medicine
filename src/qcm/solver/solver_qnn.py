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
                 criterion: nn.Module,
                 scheduler,
                 config: dict,
                 writer: SummaryWriter,
                 device: torch.device,
                 logger: logging.Logger
                 ):
        
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
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
        
        self.logger.debug(f"Starting training epoch {epoch}")
        self.train()
        losses = []
        throughput = Throughput()
        start = time.monotonic()

        for i, (x, y) in enumerate(self.dataloader):
            # Adapt label type for loss function
            if self.config['dataset_type'] == 'pcam':
                x, y = x.to(self.device), y.float().to(self.device)
                output = self(x).squeeze(1)
                loss = self.criterion(output, y.squeeze())
            else:  # tcga
                x, y = x.to(self.device), y.long().to(self.device)
                output = self(x)
                loss = self.criterion(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

            if hasattr(self.head, 'q_params'):
                q_params_tensor = self.head.q_params.data.cpu()
                current_angles = q_params_tensor.numpy()

                # Log individual angle values
                # Flatten the tensor to iterate over all angles regardless of shape (N_layers, N_qubits, 3)
                for j, angle in enumerate(current_angles.flatten()):
                    self.writer.add_scalar(
                        f'Train/Q_Angle_{j}',
                        angle,
                        epoch * len(self.dataloader) + i
                    )

                # Log the magnitude (L2-norm) of the parameter vector
                angle_magnitude = torch.linalg.norm(q_params_tensor)
                self.writer.add_scalar(
                    'Train/Q_Angle_Magnitude',
                    angle_magnitude.item(),
                    epoch * len(self.dataloader) + i
                )

            if i % 100 == 0 and i > 0:
                throughput.update(i * self.config['training']['batch_size'], time.monotonic() - start)
                self.writer.add_scalar('Train/Loss', np.mean(losses[-100:]), epoch * len(self.dataloader) + i)
                self.writer.add_scalar('Train/Throughput', throughput.compute(), epoch * len(self.dataloader) + i)
                self.logger.debug(
                    f"Epoch {epoch}, Step {i}: Loss={loss.item():.4f}, Throughput={throughput.compute():.2f} items/sec")

        self.scheduler.step()
        
        self.logger.debug(f"Finished training epoch {epoch}. Avg Loss: {np.mean(losses):.4f}")
        return np.mean(losses)
    
    def validate(self, epoch: int):
        """_summary_

        Args:
            epoch (int): _description_

        Returns:
            _type_: _description_
        """
        
        self.logger.debug(f"Starting validation for epoch {epoch}")
        self.eval()
        val_losses = []

        # Select metric based on dataset type
        if self.config['dataset_type'] == 'pcam':
            metric = BinaryF1Score().to(self.device)
            metric_name = "F1 Score"
        else:  # tcga
            num_classes = self.config['model']['num_classes']
            metric = MulticlassAccuracy(num_classes=num_classes).to(self.device)
            metric_name = "Accuracy"

        with torch.no_grad():
            for x, y in self.dataloader:
                if self.config['dataset_type'] == 'pcam':
                    x, y = x.to(self.device), y.squeeze().float().to(self.device)
                    output = self(x).squeeze(1)
                    pred = torch.sigmoid(output) > 0.5
                    val_loss = self.criterion(output, y)
                else:  # tcga
                    x, y = x.to(self.device), y.long().to(self.device)
                    output = self(x)
                    pred = torch.argmax(output, dim=1)
                    val_loss = self.criterion(output, y)

                metric.update(pred, y)
                val_losses.append(val_loss.item())

        metric_val = metric.compute()
        avg_val_loss = np.mean(val_losses)

        self.writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        self.writer.add_scalar(f'Val/{metric_name}', metric_val.item(), epoch)
        self.logger.info(f"Validation Epoch {epoch}: Loss={avg_val_loss:.4f}, {metric_name}={metric_val.item():.4f}")

        return avg_val_loss