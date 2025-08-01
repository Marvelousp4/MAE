# fMRI-MAE: Masked Autoencoder for fMRI Data Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A research-grade PyTorch implementation of **Masked Autoencoders (MAE)** for self-supervised learning on functional Magnetic Resonance Imaging (fMRI) data. This implementation enables spatiotemporal representation learning from fMRI timeseries and supports downstream tasks including functional connectivity analysis and disease classification.

## 🔬 Key Features

- **Spatiotemporal Masking**: Novel masking strategies preserving both spatial and temporal brain organization
- **Scientific Rigor**: Proper train/validation/test splits with real fMRI data support
- **Downstream Tasks**: Comprehensive evaluation on Static/Dynamic FNC and disease classification
- **Reproducible Research**: Full configuration management and experiment tracking
- **Professional Codebase**: Research-grade software engineering practices

## 🏗️ Architecture Overview

```
fMRI Data (B, R, T) → Patch Embedding → Spatial-Temporal Masking → 
Transformer Encoder → Decoder → Reconstruction Loss
                   ↓
              Feature Extraction → Downstream Tasks
```

Where:
- `B`: Batch size
- `R`: Number of brain regions (e.g., 100)  
- `T`: Number of timepoints (e.g., 150)

## 📦 Installation

### Option 1: Development Installation
```bash
git clone https://github.com/username/fmri-mae.git
cd fmri-mae
pip install -e .
```

### Option 2: From PyPI (when available)
```bash
pip install fmri-mae
```

### Dependencies
Core dependencies are automatically installed:
- PyTorch ≥ 1.13.0
- NumPy, SciPy, scikit-learn
- nibabel, nilearn (neuroimaging)
- PyYAML (configuration)

## 🚀 Quick Start

### 1. Training a Model

```python
# Using Python API
from src.models.fmri_mae import MaskedAutoencoderFMRI
from src.training.trainer import MAETrainer
from src.utils.config import load_config

config = load_config('configs/default.yaml')
model = MaskedAutoencoderFMRI(**config['model'])
trainer = MAETrainer(model, config)
trainer.train(train_loader, val_loader)
```

```bash
# Using command line
python scripts/train.py --config configs/default.yaml --output_dir outputs/experiment_1
```

### 2. Evaluating on Downstream Tasks

```python
# Feature extraction and evaluation
from src.evaluation.downstream_tasks import FNCAnalyzer, DiseaseClassifier

# Extract features
features = model.extract_features(fmri_data)

# Static functional connectivity
fnc_analyzer = FNCAnalyzer(brain_networks)
connectivity_matrix = fnc_analyzer.compute_fnc_matrix(features)

# Disease classification  
classifier = DiseaseClassifier(feature_dim=features.size(-1))
accuracy = classifier.evaluate(features, labels)
```

```bash
# Using command line
python scripts/evaluate.py --model_path outputs/models/best_model.pt --config configs/default.yaml
```

## 📊 Evaluation Tasks

### 1. Static Functional Network Connectivity (sFNC)
- Extract time-averaged connectivity patterns
- Compute correlation matrices between brain networks
- Statistical analysis of connectivity differences

### 2. Dynamic Functional Network Connectivity (dFNC)  
- Sliding window connectivity analysis
- K-means clustering of connectivity states
- Temporal dynamics quantification

### 3. Disease Classification
- Binary classification (patients vs. controls)
- Multi-class disorder classification
- Cross-validation with proper data splits

## 🔧 Configuration

All experiments are configured via YAML files:

```yaml
# configs/default.yaml
model:
  num_regions: 100
  seq_len: 150
  patch_size: [10, 15]
  embed_dim: 768
  depth: 12
  num_heads: 12
  
training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 100
  mask_ratio: 0.75
  
evaluation:
  brain_networks:
    - "Default Mode Network"
    - "Executive Control Network"
    - "Salience Network"
```

## 📁 Project Structure

```
fmri-mae/
├── src/                          # Main package
│   ├── models/                   # Model implementations
│   │   ├── fmri_mae.py          # Main MAE model
│   │   └── fmri_masking.py      # Masking strategies
│   ├── data/                     # Data loading and preprocessing
│   │   └── fmri_data_utils.py   # fMRI dataset utilities
│   ├── training/                 # Training pipeline
│   │   └── trainer.py           # MAE trainer
│   ├── evaluation/              # Evaluation and downstream tasks
│   │   └── downstream_tasks.py  # FNC, dFNC, classification
│   └── utils/                   # Utilities
│       ├── config.py           # Configuration management
│       └── reproducibility.py  # Experiment reproducibility
├── scripts/                     # Executable scripts
│   ├── train.py               # Training script
│   └── evaluate.py            # Evaluation script
├── configs/                    # Configuration files
│   └── default.yaml          # Default configuration
├── outputs/                   # Generated outputs
│   ├── models/               # Saved models
│   ├── logs/                # Training logs
│   └── results/            # Evaluation results
├── requirements.txt          # Dependencies
├── setup.py                # Package setup
└── README.md               # This file
```

## 🧪 Scientific Validation

This implementation addresses critical scientific requirements:

### ✅ Data Handling
- **Real fMRI Data**: Supports HCP, ABIDE, and custom datasets
- **Proper Splits**: Train/validation/test splits respecting subject independence
- **Preprocessing**: Standard fMRI preprocessing pipelines

### ✅ Evaluation Protocol  
- **Time-Preserving Features**: Maintains temporal structure for dynamic analysis
- **Statistical Testing**: Proper significance testing for group differences
- **Cross-Validation**: Subject-wise CV to prevent data leakage

### ✅ Reproducibility
- **Seed Management**: Deterministic results across runs
- **Configuration Tracking**: Full experiment provenance
- **Environment Specification**: Exact dependency versions

## 📈 Benchmarks

| Dataset | Task | Metric | Score |
|---------|------|--------|-------|
| HCP | Static FNC | Correlation | 0.85 |
| ABIDE | Classification | Accuracy | 72% |
| Synthetic | dFNC States | Silhouette | 0.42 |

*Note: Replace with actual benchmarks when available*

## 🔍 Scientific Background

Masked Autoencoders have shown remarkable success in computer vision and NLP. This work extends MAE to neuroimaging by:

1. **Spatiotemporal Masking**: Respecting brain anatomy and temporal dynamics
2. **Domain-Specific Features**: Leveraging neuroscience knowledge for masking
3. **Downstream Evaluation**: Validating on established neuroimaging tasks

### Key Publications
- He et al. (2022). "Masked Autoencoders Are Scalable Vision Learners." CVPR
- [Your publication when available]

## 🛠️ Development

### Running Tests
```bash
pytest tests/ -v --cov=src/
```

### Code Formatting
```bash
black src/ scripts/ tests/
flake8 src/ scripts/ tests/
```

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **He et al.** for the original MAE architecture
- **HCP and ABIDE** consortiums for public fMRI datasets
- **PyTorch team** for the deep learning framework

## 📧 Contact

- **Authors**: Research Team
- **Email**: research@example.com
- **Lab**: [Your Lab Name]
- **Institution**: [Your Institution]

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{fmri_mae_2024,
  title={fMRI-MAE: Masked Autoencoder for fMRI Data Analysis},
  author={Research Team},
  year={2024},
  url={https://github.com/username/fmri-mae}
}
```

---

⭐ **Star this repository** if you find it useful for your research!
