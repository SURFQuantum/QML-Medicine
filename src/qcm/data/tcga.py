
import os
import torch
import pickle
import numpy as np
import pandas as pd
from typing import Tuple
from torch.utils.data import Dataset


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