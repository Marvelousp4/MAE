#!/usr/bin/env python3
"""
Quick test of the improved MAE evaluation pipeline
"""

import torch
import numpy as np
from evaluate_mae_improved import ImprovedFMRIMAEEvaluator
from fmri_data_utils import load_or_generate_data
from downstream_tasks import generate_synthetic_disease_data

def quick_test():
    print("=" * 60)
    print("Quick Test of Improved fMRI MAE Pipeline")
    print("=" * 60)
    
    # Model configuration
    model_config = {
        "num_regions": 53,
        "seq_len": 200,
        "patch_size_T": 10,
        "embed_dim": 128,
        "depth": 4,
        "num_heads": 4,
        "decoder_embed_dim": 64,
        "decoder_depth": 2,
        "decoder_num_heads": 2,
        "mlp_ratio": 4.0,
        "norm_layer": torch.nn.LayerNorm
    }
    
    # Brain networks
    brain_networks = {
        "DMN": [0, 5, 10, 15, 20, 25, 30],
        "Visual": [35, 40, 45, 50],
        "Sensorimotor": [1, 6, 11, 16, 21, 26],
        "Attention": [2, 7, 12, 17, 22],
        "Frontal": [3, 8, 13, 18, 23],
        "Parietal": [4, 9, 14, 19, 24]
    }
    
    # Generate small datasets for quick testing
    print("Generating test data...")
    train_data = load_or_generate_data("quick_train_data.pt", n_samples=100, n_regions=53, n_timepoints=200)
    eval_data, labels = generate_synthetic_disease_data(n_samples=50, n_regions=53, n_timepoints=200)
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Eval data shape: {eval_data.shape}")
    print(f"Labels distribution: {labels.bincount()}")
    
    # Initialize evaluator
    evaluator = ImprovedFMRIMAEEvaluator(model_config, brain_networks, "quick_test_model.pt")
    
    # Run evaluation with small epochs for quick testing
    print("\nRunning quick evaluation...")
    results = evaluator.run_complete_evaluation(
        train_data, eval_data, labels, pretrain_epochs=10, force_retrain=True
    )
    
    # Analyze results
    print("\n" + "=" * 60)
    print("ANALYSIS OF RESULTS")
    print("=" * 60)
    
    # Check training history
    if results['pretrain_history']['loss']:
        print(f"Training loss: {results['pretrain_history']['loss'][0]:.4f} -> {results['pretrain_history']['loss'][-1]:.4f}")
    
    # Check FNC matrix
    fnc_matrix = results['fnc']['fnc_matrix']
    print(f"FNC matrix shape: {fnc_matrix.shape}")
    print(f"FNC matrix range: [{fnc_matrix.min():.4f}, {fnc_matrix.max():.4f}]")
    print(f"FNC matrix mean: {fnc_matrix.mean():.4f}")
    print(f"FNC matrix diagonal: {torch.diag(fnc_matrix).tolist()}")
    
    # Check off-diagonal correlations
    mask = ~torch.eye(fnc_matrix.size(0), dtype=bool)
    off_diag = fnc_matrix[mask]
    print(f"Off-diagonal correlations: mean={off_diag.mean():.4f}, std={off_diag.std():.4f}")
    
    # Check region features
    region_features = results['region_features']
    print(f"Region features shape: {region_features.shape}")
    print(f"Region features range: [{region_features.min():.4f}, {region_features.max():.4f}]")
    print(f"Region features mean: {region_features.mean():.4f}, std: {region_features.std():.4f}")
    
    # Check classification
    acc = results['classification']['test_metrics']['test_accuracy']
    print(f"Classification accuracy: {acc:.3f}")
    
    # Check if features are not random anymore
    if abs(region_features.mean()) < 0.1 and 0.8 < region_features.std() < 1.2:
        print("✗ WARNING: Features still look like random noise!")
    else:
        print("✓ Features show meaningful patterns (not random)")
    
    print("\n" + "=" * 60)
    print("Quick test completed!")
    print("=" * 60)

if __name__ == "__main__":
    quick_test()
