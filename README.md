# QCM

Quantum advantage for medical applications.

## Quick Start

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up the data:**
    Download and extract the dataset into the `src/data` directory.
    ```bash
    wget "https://surfdrive.surf.nl/index.php/s/mx0RdFXHHDfcY3g/download" -O pcam_tcga.tar
    tar -xf pcam_tcga.tar -C src/data/
    ```

4.  **Train the model:**
    Choose between classical and quantum mode by changing the `--mode` flag.

    * **Classical Mode:**
        ```bash
        python src/train.py --config ./configs/config.yaml --mode classical
        ```
    * **Quantum Mode:**
        ```bash
        python src/train.py --config ./configs/config.yaml --mode quantum
        ```

## Configuration (`configs/config.yaml`)

The training behavior is controlled by `config.yaml`. Here is a brief overview of the key parameters:

* **`dataset_type`**: Toggles between `'pcam'` (binary image classification) and `'tcga'` (multiclass embedding classification).
* **`model`**:
    * **`latent_dim`**: The size of the embedding vector produced by the backbone.
    * **`n_qubits`**: The number of qubits to use in the quantum head.
    * **`n_quantum_layers`**: The number of repeated layers in the quantum circuit's ansatz.
    * **`num_classes`**: Number of unique classes for the TCGA dataset.
    * **`quantum_head_type`**: The type of quantum embedding to use, either `'amplitude'` or `'angle'`.
    * **`entangling_layer`**: The type of entangling layer in the quantum circuit, either `'strong'` or `'basic'`.
* **`training`**:
    * **`batch_size`**: The number of samples per batch.
    * **`lr`**: The learning rate for the Adam optimizer.
    * **`epochs`**: The total number of training epochs.

## Expected Outputs

After running the training script, you can expect the following outputs:

* **Trained Model:** The trained model backbone will be saved in the `models/` directory (e.g., `models/model_backbone_tcga_classical_20231027_103000.pt`).
* **t-SNE Visualization:** A t-SNE visualization of the latent space will be saved in the `models/` directory (e.g., `models/latent_tsne_tcga_classical_20231027_103000.png`).
* **Training Logs:** Training and validation losses for each run are logged in `logs/training_log.csv`.
* **TensorBoard Logs:** You can view live training metrics by running:
    ```bash
    tensorboard --logdir runs
    ```
