# model/models.py

import torch
import torch.nn as nn
from torchmetrics.classification import BinaryF1Score, MulticlassAccuracy

from ..components.encoder.pcam import PCAMSpatialPreserving as PCAMBackbone
from ..components.encoder.tcga import TCGASpatialPreserving as TCGABackbone

from ..components.hamiltonian.embedding import QuantumHeadHamiltonianPaper
from ..components.hamiltonian.embedding_light import QuantumHeadHamiltonianSimple


from ..utils.image_grid import grid_shape_for_amplitude

# =============================================================================
# Main Classifier (only applies spatial-preserving backbone when head_type == 'hamiltonian')
# =============================================================================
class HybridClassifier(nn.Module):
    def __init__(self, config: dict, use_quantum: bool = False):
        super().__init__()
        
        model_cfg = config['model']
        dataset_type = config['dataset_type']
        n_qubits = model_cfg['n_qubits']
        num_classes = model_cfg['num_classes']

        latent_dim = 2 ** n_qubits
        grid_height, grid_width = grid_shape_for_amplitude(n_qubits)

        if dataset_type == 'tcga':
            self.backbone = TCGABackbone(input_dim=768, 
                                         latent_dim=latent_dim,
                                         out_grid=(grid_height, grid_width))
        else:
            self.backbone = PCAMBackbone(latent_dim=latent_dim,
                                         filters=model_cfg['pcam_filters'],
                                         out_grid=(grid_height, grid_width))
            
        variant = model_cfg.get('hamiltonian_variant', 'simple')  # 'simple' or 'paper'
        if variant == 'simple':
            self.head = QuantumHeadHamiltonianSimple(
                n_qubits=n_qubits,
                num_classes=num_classes,
                n_layers=model_cfg['n_quantum_layers'],
                entangling_layer=model_cfg['entangling_layer'],
                include_zz=model_cfg.get('hamiltonian_include_zz', True),
                time=model_cfg.get('hamiltonian_time', 1.0)
            )
            
        elif variant == 'paper':
            self.head = QuantumHeadHamiltonianPaper(
                n_qubits=n_qubits,
                num_classes=num_classes,
                n_layers=model_cfg['n_quantum_layers'],
                entangling_layer=model_cfg['entangling_layer'],
                include_xx=model_cfg.get('hamiltonian_include_xx', True),
                n_trotter_steps=model_cfg.get('hamiltonian_trotter_steps', 1),
                time=model_cfg.get('hamiltonian_time', 1.0)
            )
        else:
            raise ValueError(f"Unknown hamiltonian_variant: {variant}")

        if num_classes == 2:
            self.loss_function = nn.BCEWithLogitsLoss()
            self.validation_metric = BinaryF1Score()
        else:
            self.loss_function = nn.CrossEntropyLoss()
            self.validation_metric = MulticlassAccuracy(num_classes=num_classes)
                     

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        
        x, _ = batch
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