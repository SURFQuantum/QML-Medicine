# data/datasets.py

import os
import h5py
import torch
import pickle
import numpy as np
import pandas as pd
from typing import Tuple
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms

# =============================================================================
# PCAM Dataset (Original)
# =============================================================================
class PCAMDataset(Dataset):
    def __init__(self, x_path: str, y_path: str, transform=None):
        self.x_path = x_path
        self.y_path = y_path
        self.transform = transform

        self.x_data = h5py.File(x_path, 'r')['x']
        self.y_data = h5py.File(y_path, 'r')['y']

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        image = self.x_data[idx]
        label = self.y_data[idx][0]
        image = image.astype(np.uint8)

        if self.transform:
            image = self.transform(image)
            
        # PCAM is binary classification, requires float for BCEWithLogitsLoss
        return image, torch.tensor(label).float()

# =============================================================================
# TCGA Dataset (New)
# =============================================================================
class TCGADataset(Dataset):
    def __init__(self, pickle_path: str, csv_path: str):
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Load data
        with open(pickle_path, 'rb') as f:
            self.embedding_data = pickle.load(f)
        self.labels_df = pd.read_csv(csv_path)

        # Create label mapping from string to integer
        self.cancer_types = sorted(self.labels_df['cancer_type'].unique())
        self.class_to_idx = {name: i for i, name in enumerate(self.cancer_types)}
        self.idx_to_class = {i: name for i, name in enumerate(self.cancer_types)}
        
        self.labels_df['label_idx'] = self.labels_df['cancer_type'].map(self.class_to_idx)

        # Filter for patients present in both files and with non-empty embeddings
        embedding_keys = set(self.embedding_data.keys())
        label_keys = set(self.labels_df['patient_id'])
        self.valid_patient_ids = sorted(list(embedding_keys.intersection(label_keys)))
        
        # Further filter out patients with empty embeddings
        self.valid_patient_ids = [
            pid for pid in self.valid_patient_ids 
            if "embeddings" in self.embedding_data[pid] and len(self.embedding_data[pid]["embeddings"]) > 0
        ]
        
        # Align labels_df with valid patient IDs for easy lookup
        self.labels_df.set_index('patient_id', inplace=True)

    def __len__(self):
        return len(self.valid_patient_ids)

    def __getitem__(self, idx):
        patient_id = self.valid_patient_ids[idx]
        
        # Get embeddings (taking the mean to get a single vector per patient)
        embeddings = np.array(self.embedding_data[patient_id]["embeddings"])
        embedding_vector = torch.tensor(embeddings.mean(axis=0)).float()

        # Get label
        label_idx = self.labels_df.loc[patient_id, 'label_idx']
        
        # TCGA is multi-class classification, requires long for CrossEntropyLoss
        return embedding_vector, torch.tensor(label_idx).long()

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