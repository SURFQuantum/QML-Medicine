from pathlib import Path
import logging
import yaml
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

def log_losses_to_csv(filepath: Path, run_name: str, train_losses: list, val_losses: list):
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
    
    # Add the new loss data as columns
    df[train_col_name] = train_losses
    df[val_col_name] = val_losses

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
    print(labels)
    tsne = TSNE(n_components=2, perplexity=30)
    embedded = tsne.fit_transform(latents)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embedded[:, 0], y=embedded[:, 1], hue=labels, palette='coolwarm', s=10)
    plt.title("t-SNE of Latent Space")
    plt.savefig(save_path)
    plt.close()

def plot_all_runs_losses(all_runs_losses: dict, dataset_type: str, timestamp: str, save_path: Path):
    """
    Plots the training and validation loss curves for all configurations (0 to max_num_of_quantum layers).
    :param all_runs_losses: Dictionary {num_quantum_layers: (train_losses, val_losses)}
    :param dataset_type: Name of the dataset for the title.
    :param timestamp: Timestamp string for unique saving.
    :param save_path: Path to save the plot.
    """
    if not all_runs_losses:
        logger.warning("No losses recorded to plot.")
        return

    logger.info(f"Generating aggregate loss plot and saving to {save_path}")

    num_epochs = len(next(iter(all_runs_losses.values()))[0])
    epochs = range(1, num_epochs + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_runs_losses)))
    sorted_keys = sorted(all_runs_losses.keys())

    for i, n_layers in enumerate(sorted_keys):
        train_losses, val_losses = all_runs_losses[n_layers]
        label_base = f'{n_layers} Q Layers'
        if n_layers == 0:
            label_base = 'Classical (0 Q Layers)'

        ax.plot(epochs, train_losses,
                label=f'{label_base} - Train Loss',
                linestyle='-',
                linewidth=1.5,
                color=colors[i])

        ax.plot(epochs, val_losses,
                label=f'{label_base} - Val Loss',
                linestyle='--',
                linewidth=2.0,
                color=colors[i])

    ax.set_title(f'Hybrid Quantum-Classical Model Performance on {dataset_type.upper()} Dataset', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(loc='upper right', ncol=2, fontsize=8)
    ax.grid(True, which='both', linestyle='-')

    try:
        plt.savefig(save_path)
        logger.info(f"Aggregate loss plot successfully saved to: {save_path}")
    except Exception as e:
        logger.error(f"Error saving plot: {e}")
    finally:
        plt.close(fig)

def load_config(path: str) -> dict:
    logger.debug(f"Loading config from: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)