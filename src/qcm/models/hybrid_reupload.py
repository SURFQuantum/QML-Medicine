import torch
import torch.nn as nn
from torchmetrics.classification import BinaryF1Score, MulticlassAccuracy
from ..components.encoder.pcam import PCAM as PCAMBackbone
from ..components.encoder.tcga import TCGA as TCGABackbone
from ..components.qnn.classical import ClassicalHead
from ..components.reupload.reupload import QuantumHeadReupload

class HybridReuploadClassifier(nn.Module):

    def __init__(self, config: dict, use_quantum: bool = False):
        super().__init__()
        model_cfg = config['model']
        dataset_type = config['dataset_type']
        
        # 1. Select Backbone
        if dataset_type == 'pcam':
            self.backbone = PCAMBackbone(
                latent_dim=model_cfg['latent_dim'], 
                filters=model_cfg['pcam_filters']
            )
            num_classes = 2
        elif dataset_type == 'tcga':
            self.backbone = TCGABackbone(
                input_dim = 768, 
                latent_dim=model_cfg['latent_dim']
            )
            num_classes = model_cfg['num_classes']
        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

        # 2. Select Head
        if use_quantum:
            head_type = model_cfg.get('quantum_head_type', 'reupload')

            if head_type == 'reupload':
                self.head = QuantumHeadReupload(
                    num_qubits = model_cfg['n_qubits'],
                    num_features = model_cfg['latent_dim'],
                    num_classes=num_classes,
                    n_repetitions=model_cfg['n_quantum_layers'],
                    entangling_layer=model_cfg['entangling_layer']
                )
            else:
                raise ValueError(f"Unknown quantum_head_type: {head_type}")
        else:
            latent_dim = model_cfg['latent_dim']
            self.head = ClassicalHead(latent_dim, num_classes)
    
        self.normalisation = torch.nn.Sigmoid()

        self.validation_metric = self.configure_validatiaon_metric(num_classes)
        self.optimizer = self.configure_optimizer()


    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.normalisation(x)
        return self.head(x, y)
    
    def configure_validatiaon_metric(self, num_classes: int):
        if num_classes == 2:
            return BinaryF1Score()
        else:
            return MulticlassAccuracy(num_classes=num_classes)
    
    def configure_optimizer(self, optimizer: torch.optim.Optimizer = None):
        """Setup the optimizer and learning rate scheduler."""
        if optimizer is None:
            return torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), 
            )
        else:
            return optimizer

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        """A single trainig step over a batch of data.

        Args:
            batch (torch.Tensor): data batch

        Returns:
            torch.Tensor: loss value
        """
        x, y = batch
        loss = self.forward(x, y).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
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
        self.valifation_metric.update(pred, y)
        return loss, pred

    def predict(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Predict the class probabilities for input data.

        Args:
            x (torch.Tensor): input data

        Returns:
            torch.Tensor: predicted class probabilities
        """
        nbatch = x.shape[0]
        nstate = self.head.num_classes
        target_labels = self.head.target_labels.reshape(1, -1)
        x = torch.repeat_interleave(x, nstate, dim=0)
        y = torch.repeat_interleave(target_labels, nbatch, dim=0).reshape(-1 ,1)
        output = self(x, y).reshape(-1, nstate)
        val_loss, pred = output.min(dim=1) 
        return val_loss, pred

class ReuploadClassifier(HybridReuploadClassifier):
    def __init__(self, config: dict, use_quantum: bool = False):
        super().__init__()
        model_cfg = config['model']

        num_classes = 2

        self.head = QuantumHeadReupload(
            num_qubits = model_cfg['n_qubits'], 
            num_features = model_cfg['latent_dim'],
            num_classes=num_classes,
            n_repetitions=model_cfg['n_quantum_layers'],
            entangling_layer=model_cfg['entangling_layer']
        )    

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.head(x, y)