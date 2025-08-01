#!/usr/bin/env python3
"""
Evaluate trained fMRI MAE model on downstream tasks.

Usage:
    python scripts/evaluate.py --model_path outputs/models/final_model.pt --config configs/default.yaml
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
import json
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.fmri_mae import MaskedAutoencoderFMRI
from src.evaluation.downstream_tasks import FNCAnalyzer, DynamicFNCAnalyzer, DiseaseClassifier
from src.evaluation.downstream_tasks import generate_synthetic_disease_data
from src.utils.config import load_config
import src.utils.reproducibility as repro


class FeatureExtractor:
    """Extract features from trained MAE model."""
    
    def __init__(self, model: MaskedAutoencoderFMRI):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def extract_features(self, data: torch.Tensor) -> dict:
        """Extract features preserving temporal structure."""
        data = data.to(self.device)
        
        with torch.no_grad():
            features, _, _ = self.model.forward_encoder(data, mask_ratio=0.0)
            
        N, total_patches, embed_dim = features.shape
        
        # Separate CLS token
        cls_features = features[:, 0, :].cpu()
        patch_features = features[:, 1:, :].cpu()
        
        # Organize by regions
        num_regions = self.model.num_regions
        patches_per_region = patch_features.size(1) // num_regions
        
        region_features = patch_features[:, :num_regions*patches_per_region, :]
        region_features = region_features.view(N, num_regions, patches_per_region, embed_dim)
        
        # Static features (time-averaged)
        static_features = region_features.mean(dim=2)
        
        return {
            'cls_features': cls_features,
            'patch_features': patch_features,
            'region_features': region_features,
            'static_features': static_features
        }


def evaluate_downstream_tasks(features: dict, config: dict, data: torch.Tensor, 
                            labels: torch.Tensor) -> dict:
    """Evaluate all downstream tasks."""
    results = {}
    
    brain_networks = config['evaluation']['brain_networks']
    
    # 1. Static FNC Analysis
    print("Evaluating static functional connectivity...")
    fnc_analyzer = FNCAnalyzer(brain_networks)
    network_features = fnc_analyzer.extract_network_features(features['static_features'])
    fnc_matrix = fnc_analyzer.compute_fnc_matrix(network_features)
    
    results['fnc'] = {
        'matrix': fnc_matrix,
        'network_features': network_features
    }
    
    # 2. Dynamic FNC Analysis
    print("Evaluating dynamic functional connectivity...")
    dfnc_analyzer = DynamicFNCAnalyzer(brain_networks)
    dynamic_features = dfnc_analyzer.extract_dynamic_features(data)
    state_labels, state_centroids = dfnc_analyzer.cluster_dynamic_states(
        dynamic_features, n_states=4
    )
    
    results['dfnc'] = {
        'state_labels': state_labels,
        'state_centroids': state_centroids,
        'num_windows': len(dynamic_features)
    }
    
    # 3. Disease Classification
    print("Evaluating disease classification...")
    classifier = DiseaseClassifier(
        feature_dim=features['cls_features'].size(-1),
        num_classes=2
    )
    
    # Split data for evaluation
    n_test = int(0.3 * len(features['cls_features']))
    indices = torch.randperm(len(features['cls_features']))
    
    train_indices = indices[n_test:]
    test_indices = indices[:n_test]
    
    train_features = features['cls_features'][train_indices]
    train_labels = labels[train_indices]
    test_features = features['cls_features'][test_indices]
    test_labels = labels[test_indices]
    
    # Train and evaluate
    train_metrics = classifier.train_classifier(train_features, train_labels)
    test_metrics = classifier.evaluate(test_features, test_labels)
    
    results['classification'] = {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained fMRI MAE model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='outputs/results',
                       help='Output directory for results')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to evaluation data (if not provided, uses synthetic)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup reproducibility
    repro.set_seed(config['seed'])
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("fMRI MAE Evaluation")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output_dir}")
    
    # Load model
    print("\nLoading trained model...")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    model = MaskedAutoencoderFMRI(**config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded successfully")
    print(f"Training history available: {'history' in checkpoint}")
    
    # Load evaluation data
    print("\nLoading evaluation data...")
    if args.data_path and Path(args.data_path).exists():
        eval_data = torch.load(args.data_path)
        # Assume labels are provided separately or embedded
        labels = torch.randint(0, 2, (len(eval_data),))  # Placeholder
    else:
        print("Using synthetic evaluation data...")
        eval_data, labels = generate_synthetic_disease_data(
            n_samples=200, n_regions=config['model']['num_regions'], 
            n_timepoints=config['model']['seq_len']
        )
    
    print(f"Evaluation data: {eval_data.shape}")
    print(f"Labels distribution: {labels.bincount()}")
    
    # Extract features
    print("\nExtracting features...")
    feature_extractor = FeatureExtractor(model)
    features = feature_extractor.extract_features(eval_data)
    
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")
    
    # Evaluate downstream tasks
    print("\nEvaluating downstream tasks...")
    results = evaluate_downstream_tasks(features, config, eval_data, labels)
    
    # Analyze and save results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # FNC Analysis
    fnc_matrix = results['fnc']['matrix']
    print(f"\n📊 Static Connectivity:")
    print(f"  FNC matrix shape: {fnc_matrix.shape}")
    print(f"  Mean connectivity: {fnc_matrix.mean():.4f}")
    print(f"  Connectivity range: [{fnc_matrix.min():.3f}, {fnc_matrix.max():.3f}]")
    
    # Dynamic FNC
    print(f"\n🔄 Dynamic Connectivity:")
    print(f"  Number of time windows: {results['dfnc']['num_windows']}")
    print(f"  Number of states identified: {len(torch.unique(results['dfnc']['state_labels']))}")
    
    # Classification
    test_acc = results['classification']['test_metrics']['test_accuracy']
    print(f"\n🎯 Classification:")
    print(f"  Test accuracy: {test_acc:.3f}")
    
    # Feature quality assessment
    static_feat = features['static_features']
    print(f"\n🧠 Feature Quality:")
    print(f"  Feature mean: {static_feat.mean():.4f}")
    print(f"  Feature std: {static_feat.std():.4f}")
    print(f"  Feature range: [{static_feat.min():.3f}, {static_feat.max():.3f}]")
    
    # Save results (simplified to avoid JSON issues)
    results_file = output_dir / 'evaluation_results.txt'
    
    with open(results_file, 'w') as f:
        f.write("fMRI-MAE Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        
        # Write key metrics
        fnc_matrix = results['fnc']['matrix']
        f.write(f"Static Connectivity:\n")
        f.write(f"  FNC matrix shape: {fnc_matrix.shape}\n")
        f.write(f"  Mean connectivity: {fnc_matrix.mean():.4f}\n")
        f.write(f"  Connectivity range: [{fnc_matrix.min():.3f}, {fnc_matrix.max():.3f}]\n\n")
        
        f.write(f"Dynamic Connectivity:\n")
        f.write(f"  Number of time windows: {results['dfnc']['num_windows']}\n")
        f.write(f"  Number of states: {len(torch.unique(results['dfnc']['state_labels']))}\n\n")
        
        test_acc = results['classification']['test_metrics']['test_accuracy']
        f.write(f"Classification:\n")
        f.write(f"  Test accuracy: {test_acc:.3f}\n\n")
        
        static_feat = features['static_features']
        f.write(f"Feature Quality:\n")
        f.write(f"  Feature mean: {static_feat.mean():.4f}\n")
        f.write(f"  Feature std: {static_feat.std():.4f}\n")
        f.write(f"  Feature range: [{static_feat.min():.3f}, {static_feat.max():.3f}]\n")
    
    print(f"\nResults saved to: {results_file}")
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
