## Getting Started

### 1\. Installation

Clone the repository and install the package in editable mode. Dependencies will be installed automatically.

```bash
git clone https://github.com/your-username/fmri-mae.git
cd fmri-mae
pip install -e .
```

### 2\. Configuration

All model, training, and evaluation parameters are managed in YAML configuration files located in the `configs/` directory. Modify `configs/default.yaml` to suit your experiment.

```yaml
# configs/default.yaml
model:
  num_regions: 100
  seq_len: 150
  patch_size: [10, 15] # [region_patch_size, time_patch_size]
  embed_dim: 768
  # ... other model params

training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 100
  mask_ratio: 0.75
  # ... other training params
```

### 3\. Training

Run the training script with your chosen configuration file. Model checkpoints and logs will be saved to the specified output directory.

```bash
python scripts/train.py --config configs/default.yaml --output_dir outputs/your_experiment_name
```

### 4\. Evaluation

Evaluate a trained model on downstream tasks (sFNC, dFNC, classification) using the evaluation script.

```bash
python scripts/evaluate.py --model_path outputs/your_experiment_name/models/best_model.pt --config configs/default.yaml
```

-----

## Methodology

This implementation extends the MAE framework to the neuroimaging domain with a focus on scientific rigor.

  * **Spatiotemporal MAE**: The core model is a Transformer-based autoencoder. It learns to reconstruct heavily masked (`75%`) fMRI signals, forcing it to learn meaningful representations of brain activity patterns. The input fMRI data is structured as `(Batch, Regions, Time)`, and the model uses spatiotemporal "patches" as its fundamental units.
  * **Scientific Validation**: The framework ensures subject independence in all data splits (train/validation/test) to prevent data leakage. It supports standard fMRI datasets (e.g., HCP, ABIDE) and provides robust evaluation protocols for established neuroimaging tasks.
  * **Reproducibility**: Experiments are fully reproducible through comprehensive configuration management and controlled random seeding.

