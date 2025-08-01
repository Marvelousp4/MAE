#!/usr/bin/env python3
"""
Train fMRI MAE model with scientific rigor.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --output_dir outputs/experiment_1
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.fmri_mae import MaskedAutoencoderFMRI
from src.data.fmri_data_utils import load_or_generate_data
from src.training.trainer import MAETrainer
from src.utils.config import load_config, save_config
import src.utils.reproducibility as repro


def main():
    parser = argparse.ArgumentParser(description='Train fMRI MAE model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.output_dir:
        config['output']['checkpoint_dir'] = str(Path(args.output_dir) / 'models')
        config['output']['log_dir'] = str(Path(args.output_dir) / 'logs')
        config['output']['results_dir'] = str(Path(args.output_dir) / 'results')
    
    if args.seed:
        config['seed'] = args.seed
    
    # Setup reproducibility
    repro.set_seed(config['seed'])
    if config.get('deterministic', True):
        repro.set_deterministic()
    
    # Create output directories
    for output_type in ['checkpoint_dir', 'log_dir', 'results_dir']:
        Path(config['output'][output_type]).mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    save_config(config, Path(config['output']['results_dir']) / 'config.yaml')
    
    print("=" * 80)
    print("fMRI MAE Training")
    print("=" * 80)
    print(f"Configuration: {args.config}")
    print(f"Output directory: {config['output']['checkpoint_dir']}")
    print(f"Seed: {config['seed']}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Load data
    print("\nLoading training data...")
    train_data = load_or_generate_data(
        'data/train_data.pt',
        n_samples=config['data']['n_pretrain_subjects'],
        n_regions=config['model']['num_regions'],
        n_timepoints=config['model']['seq_len']
    )
    
    # Split train/validation
    split_idx = int(config['data']['train_val_split'] * len(train_data))
    val_data = train_data[split_idx:]
    train_data = train_data[:split_idx]
    
    print(f"Train data: {train_data.shape}")
    print(f"Validation data: {val_data.shape}")
    
    # Create model
    print("\nCreating model...")
    model = MaskedAutoencoderFMRI(**config['model'])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Create trainer
    trainer_config = {**config['training'], **config['output']}
    trainer = MAETrainer(model, trainer_config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.history = checkpoint.get('history', trainer.history)
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(train_data, val_data, epochs=config['training']['epochs'])
    
    # Save final model
    final_model_path = Path(config['output']['checkpoint_dir']) / 'final_model.pt'
    trainer.save_final_model(str(final_model_path))
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {history['best_val_loss']:.4f}")
    print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
