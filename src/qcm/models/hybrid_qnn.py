# model/models.py

import torch
import torch.nn as nn
from torchmetrics.classification import BinaryF1Score, MulticlassAccuracy

from ..components.encoder.pcam import PCAM as PCAMBackbone
from ..components.encoder.tcga import TCGA as TCGABackbone

from ..components.qnn.classical import ClassicalHead
from ..components.qnn.amplitude_encoding import QuantumHeadAmplitude
from ..components.qnn.angle_encoding import QuantumHeadAngle


class HybridClassifier(nn.Module):

    def __init__(self, config: dict, use_quantum: bool = False, num_quantum_layers: int = 1):
        super().__init__()
        model_cfg = config['model']
        dataset_type = config['dataset_type']
        self.num_quantum_layers = num_quantum_layers
        
        # 1. Select Backbone
        if dataset_type == 'pcam':
            self.backbone = PCAMBackbone(
                latent_dim=model_cfg['latent_dim'], 
                filters=model_cfg['pcam_filters']
            )
            num_classes = 1
        elif dataset_type == 'tcga':
            self.backbone = TCGABackbone(
                input_dim=768, 
                latent_dim=model_cfg['latent_dim']
            )
            num_classes = model_cfg['num_classes']
        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

        # 2. Select Head
        if use_quantum:
            head_type = model_cfg.get('quantum_head_type', 'amplitude')
            n_qubits = model_cfg['n_qubits']
            
            if head_type == 'amplitude':
                latent_dim = 2**n_qubits
                if dataset_type == 'tcga':
                     self.backbone = TCGABackbone(input_dim=768, latent_dim=latent_dim)
                else:
                     self.backbone = PCAMBackbone(latent_dim=latent_dim, filters=model_cfg['pcam_filters'])

                self.head = QuantumHeadAmplitude(
                    n_qubits=n_qubits,
                    num_classes=num_classes,
                    n_layers=self.num_quantum_layers,
                    entangling_layer=model_cfg['entangling_layer']
                )
            elif head_type == 'angle':
                latent_dim = n_qubits
                if dataset_type == 'tcga':
                     self.backbone = TCGABackbone(input_dim=768, latent_dim=latent_dim)
                else:
                     self.backbone = PCAMBackbone(latent_dim=latent_dim, filters=model_cfg['pcam_filters'])

                self.head = QuantumHeadAngle(
                    n_qubits=n_qubits,
                    num_classes=num_classes,
                    n_layers=self.num_quantum_layers,
                    entangling_layer=model_cfg['entangling_layer']
                )
            else:
                raise ValueError(f"Unknown quantum_head_type: {head_type}")
        else:
            latent_dim = model_cfg['latent_dim']
            self.head = ClassicalHead(latent_dim, num_classes)

        if num_classes == 2:
            self.loss_function = nn.BCEWithLogitsLoss()
            self.validation_metric = BinaryF1Score()
        else:
            self.loss_function = nn.CrossEntropyLoss()
            self.validation_metric = MulticlassAccuracy(num_classes=num_classes)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward function of the module.

        Args:
            x (torch.Tensor): input data

        Returns:
            torch.Tensor: output of the module
        """
        x = self.backbone(x)
        return self.head(x)
    
    def training_step(self, batch: torch.Tensor, 
                      optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """A single training step over a batch of data.

        Args:
            batch (torch.Tensor): data batch
            optimizer (torch.optim.Optimizer): optimizer to use

        Returns:
            torch.Tensor: loss value
        """
        
        x, y = batch
        output = self.forward(x).mean()
        loss = self.loss_function(output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss
    
    def validation_step(self, batch: torch.Tensor) -> torch.Tensor:
        """A single validation step over a batch of data.

        Args:
            batch (torch.Tensor): data batch

        Returns:
            torch.Tensor: loss value
        """
        x, y = batch
        loss, pred = self.predict(x, y)
        self.validation_metric.update(pred, y)
        return loss, pred
    
    def predict(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Predict the class probabilities for input data.

        Args:
            x (torch.Tensor): input data
            y (torch.Tensor): true labels

        Returns:
            torch.Tensor: loss value
            torch.Tensor: predicted labels
        """

        if self.num_classes == 2:
            x, y = x, y.squeeze().float()
            output = self.forward(x).squeeze(1)
            pred = torch.sigmoid(output) > 0.5
            val_loss = self.loss_function(output, y)
        else: 
            x, y = x, y.long()
            output = self.forward(x)
            pred = torch.argmax(output, dim=1)
            val_loss = self.loss_function(output, y)
        
        return val_loss, pred