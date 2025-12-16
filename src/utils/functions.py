from pathlib import Path
import logging
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import Tuple, Any
import torch
import seaborn as sns
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def log_losses_to_csv(filepath: Path, run_name: str, train_losses: list, val_losses: list, accuracies: list):
    """
    Logs training and validation losses to a CSV file, adding a new column for each run.
    """
    try:
        # Read existing log file
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        # Create a new DataFrame if the file doesn't exist
        num_epochs = len(train_losses)
        df = pd.DataFrame({'Epoch': range(num_epochs)})

    # Define new column names for this run
    train_col_name = f'train_loss_{run_name}'
    val_col_name = f'val_loss_{run_name}'
    acc_col_name = f'accuracy_{run_name}'

    # Add the new loss data as columns
    df[train_col_name] = train_losses
    df[val_col_name] = val_losses
    df[acc_col_name] = accuracies

    # Save the updated DataFrame back to CSV
    filepath.parent.mkdir(parents=True, exist_ok=True) # Ensure the directory exists
    df.to_csv(filepath, index=False)
    logger.info(f"Losses for run '{run_name}' have been logged to {filepath}")

def visualize_latents(model: torch.nn.Module, dataloader: DataLoader, save_path: str, device: str):
    model.eval()
    latents = []
    labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.squeeze()
            feats = model.backbone(x)
            latents.append(feats.view(feats.size(0), -1).cpu())
            labels.extend(y.cpu().tolist())

    latents = torch.cat(latents)
    tsne = TSNE(n_components=2, perplexity=30)
    embedded = tsne.fit_transform(latents)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embedded[:, 0], y=embedded[:, 1], hue=labels, palette='coolwarm', s=10)
    plt.title("t-SNE of Latent Space")
    plt.savefig(save_path)
    plt.close()
