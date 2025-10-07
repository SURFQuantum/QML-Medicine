# data/datasets.py

import os
from typing import Tuple
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from .pcam import PCAMDataset
from .tcga import TCGADataset

# =============================================================================
# Unified Dataloader Function
# =============================================================================
def get_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    dataset_type = config.get('dataset_type', 'pcam') # Default to pcam if not specified

    if dataset_type == 'pcam':
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_val = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        base = config["pcam_data"]["path"]
        train_x = os.path.join(base, "camelyonpatch_level_2", "camelyonpatch_level_2_split_train_x.h5")
        train_y = os.path.join(base, "camelyonpatch_level_2", "camelyonpatch_level_2_split_train_y.h5")
        val_x = os.path.join(base, "camelyonpatch_level_2", "camelyonpatch_level_2_split_valid_x.h5")
        val_y = os.path.join(base, "camelyonpatch_level_2", "camelyonpatch_level_2_split_valid_y.h5")

        train_dataset = PCAMDataset(train_x, train_y, transform=transform_train)
        val_dataset = PCAMDataset(val_x, val_y, transform=transform_val)
        
    elif dataset_type == 'tcga':
        full_dataset = TCGADataset(
            pickle_path=config['tcga_data']['pickle_path'],
            csv_path=config['tcga_data']['csv_path']
        )
        
        # Create train/validation split
        val_split = config['tcga_data']['val_split']
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    else:
        raise ValueError(f"Invalid dataset_type '{dataset_type}' in config. Must be 'pcam' or 'tcga'.")

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['num_workers'])

    return train_loader, val_loader