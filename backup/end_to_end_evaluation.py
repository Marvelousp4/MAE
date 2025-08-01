"""
Complete End-to-End Evaluation for fMRI MAE

This script performs the complete evaluation pipeline:
1. Pretrains MAE on fMRI data
2. Extracts features using the pretrained encoder
3. Evaluates on downstream tasks using the extracted features
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt

from fmri_mae import mae_fmri_base
from fmri_data_utils import generate_synthetic_fmri_data
from downstream_tasks import (
    FNCAnalyzer, 
    DynamicFNCAnalyzer, 
    DiseaseClassifier, 
    generate_synthetic_disease_data
)


class EndToEndEvaluator:
    """Complete end-to-end evaluation pipeline"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.brain_networks = {
            "DMN": [0, 5, 10, 15, 20],
            "Visual": [25, 30, 35, 40],
            "Sensorimotor": [1, 6, 11, 16],
            "Attention": [2, 7, 12, 17],
            "Frontal": [3, 8, 13, 18]
        }
        
    def pretrain_mae(self, data: torch.Tensor, epochs: int = 20) -> Tuple[torch.nn.Module, Dict]:
        """
        Pretrain MAE model
        
        Args:
            data: Training data [N, P, T]
            epochs: Number of training epochs
            
        Returns:
            Trained model and training history
        """
        print("=" * 50)
        print("Step 1: Pretraining MAE")
        print("=" * 50)
        
        # Create model
        model = mae_fmri_base(num_regions=53, seq_len=200, patch_size_T=20)
        model = model.to(self.device)
        data = data.to(self.device)
        
        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
        
        history = {"loss": []}
        model.train()
        
        print(f"Training on {self.device} with {data.size(0)} samples")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # Simple batch processing
            batch_size = 32
            n_batches = (data.size(0) + batch_size - 1) // batch_size
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, data.size(0))
                batch_data = data[start_idx:end_idx]
                
                # Forward pass
                optimizer.zero_grad()
                loss, pred, mask = model(batch_data, mask_ratio=0.75)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            history["loss"].append(avg_loss)
            
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1:2d}/{epochs}: Loss = {avg_loss:.4f}")
        
        print("MAE pretraining completed!")
        return model, history
    
    def extract_features(self, model: torch.nn.Module, data: torch.Tensor) -> torch.Tensor:
        """
        Extract features using pretrained MAE encoder
        
        Args:
            model: Pretrained MAE model
            data: Input data [N, P, T]
            
        Returns:
            Extracted features [N, L_visible+1, D]
        """
        print("\n" + "=" * 50)
        print("Step 2: Extracting Features")
        print("=" * 50)
        
        model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            # Use encoder to extract features (no masking for feature extraction)
            features, _, _ = model.forward_encoder(data, mask_ratio=0.0)
            
        print(f"Extracted features shape: {features.shape}")
        print(f"Feature dimension: {features.shape[-1]}")
        
        return features
    
    def evaluate_downstream_tasks(self, features: torch.Tensor, labels: torch.Tensor, 
                                 original_data: torch.Tensor) -> Dict:
        """
        Evaluate downstream tasks using extracted features
        
        Args:
            features: Extracted features [N, L, D]
            labels: Disease labels [N]
            original_data: Original fMRI data for dFNC [N, P, T]
            
        Returns:
            Downstream evaluation results
        """
        print("\n" + "=" * 50)
        print("Step 3: Downstream Task Evaluation")
        print("=" * 50)
        
        results = {}
        
        # Convert features to region-level for compatibility
        # Remove CLS token and reshape to [N, P, D]
        features_no_cls = features[:, 1:, :]  # Remove CLS token
        N, L, D = features_no_cls.shape
        
        # For spatiotemporal patches: L = 530 patches, need to convert to 53 regions
        patches_per_region = L // 53
        region_features = features_no_cls.view(N, 53, patches_per_region, D).mean(dim=2)  # [N, 53, D]
        
        print(f"Region-level features shape: {region_features.shape}")
        
        # 1. FNC Analysis
        print("\nEvaluating FNC...")
        fnc_analyzer = FNCAnalyzer(self.brain_networks)
        network_features = fnc_analyzer.extract_network_features(region_features)
        fnc_matrix = fnc_analyzer.compute_fnc_matrix(network_features)
        
        results["fnc"] = {
            "matrix": fnc_matrix,
            "network_features": network_features
        }
        print(f"FNC matrix computed: {fnc_matrix.shape}")
        
        # 2. Dynamic FNC Analysis
        print("\nEvaluating dynamic FNC...")
        dfnc_analyzer = DynamicFNCAnalyzer(self.brain_networks)
        dynamic_features = dfnc_analyzer.extract_dynamic_features(original_data.cpu())
        state_labels, state_centroids = dfnc_analyzer.cluster_dynamic_states(
            dynamic_features, n_states=3
        )
        
        results["dfnc"] = {
            "state_labels": state_labels,
            "state_centroids": state_centroids,
            "num_windows": len(dynamic_features)
        }
        print(f"Dynamic states identified: {len(torch.unique(state_labels))}")
        
        # 3. Disease Classification using extracted features
        print("\nEvaluating disease classification...")
        
        # Global average pooling of features for classification
        pooled_features = region_features.mean(dim=1)  # [N, D]
        
        # Split data
        n_test = len(pooled_features) // 3
        train_features = pooled_features[n_test:].cpu()
        train_labels = labels[n_test:].cpu()
        test_features = pooled_features[:n_test].cpu()
        test_labels = labels[:n_test].cpu()
        
        # Train classifier
        classifier = DiseaseClassifier(feature_dim=pooled_features.shape[1], num_classes=2)
        train_metrics = classifier.train_classifier(train_features, train_labels)
        test_metrics = classifier.evaluate(test_features, test_labels)
        
        results["classification"] = {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics
        }
        
        print(f"Train accuracy: {train_metrics['train_accuracy']:.3f}")
        print(f"Test accuracy: {test_metrics['test_accuracy']:.3f}")
        
        return results
    
    def run_complete_evaluation(self, data: torch.Tensor, labels: torch.Tensor, 
                              epochs: int = 20) -> Dict:
        """
        Run complete end-to-end evaluation
        
        Args:
            data: fMRI data [N, P, T]
            labels: Disease labels [N]
            epochs: Pretraining epochs
            
        Returns:
            Complete evaluation results
        """
        print("Starting Complete End-to-End Evaluation")
        print("=" * 60)
        
        # Step 1: Pretrain MAE
        model, pretrain_history = self.pretrain_mae(data, epochs=epochs)
        
        # Step 2: Extract features using pretrained encoder
        features = self.extract_features(model, data)
        
        # Step 3: Evaluate downstream tasks
        downstream_results = self.evaluate_downstream_tasks(features, labels, data)
        
        # Combine results
        results = {
            "pretrain_history": pretrain_history,
            "features": features,
            "downstream": downstream_results
        }
        
        print("\n" + "=" * 60)
        print("Complete evaluation finished!")
        print("=" * 60)
        
        return results
    
    def visualize_results(self, results: Dict):
        """Visualize evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Training loss
        ax1 = axes[0, 0]
        ax1.plot(results["pretrain_history"]["loss"])
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("MAE Pretraining Loss")
        ax1.grid(True)
        
        # Plot 2: FNC matrix
        ax2 = axes[0, 1]
        fnc_matrix = results["downstream"]["fnc"]["matrix"].cpu().numpy()
        im = ax2.imshow(fnc_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax2.set_title("Functional Network Connectivity")
        plt.colorbar(im, ax=ax2)
        
        # Plot 3: Dynamic states distribution
        ax3 = axes[1, 0]
        state_labels = results["downstream"]["dfnc"]["state_labels"].cpu().numpy()
        ax3.hist(state_labels, bins=len(np.unique(state_labels)), alpha=0.7)
        ax3.set_xlabel("Dynamic State")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Dynamic Connectivity States")
        
        # Plot 4: Feature distribution
        ax4 = axes[1, 1]
        features = results["features"][:, 0, :].cpu().numpy()  # CLS token features
        ax4.hist(features.flatten(), bins=50, alpha=0.7)
        ax4.set_xlabel("Feature Value")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Extracted Feature Distribution")
        
        plt.tight_layout()
        plt.show()


def main():
    """Main evaluation function"""
    print("fMRI MAE: Complete End-to-End Evaluation")
    print("=" * 60)
    
    # Generate data
    print("Generating synthetic fMRI data...")
    fmri_data, disease_labels = generate_synthetic_disease_data(
        n_samples=200, n_regions=53, n_timepoints=200
    )
    
    print(f"Data shape: {fmri_data.shape}")
    print(f"Label distribution: {disease_labels.bincount()}")
    
    # Run evaluation
    evaluator = EndToEndEvaluator()
    results = evaluator.run_complete_evaluation(
        fmri_data, disease_labels, epochs=15
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    final_loss = results["pretrain_history"]["loss"][-1]
    test_acc = results["downstream"]["classification"]["test_metrics"]["test_accuracy"]
    n_states = len(torch.unique(results["downstream"]["dfnc"]["state_labels"]))
    
    print(f"Final pretraining loss: {final_loss:.4f}")
    print(f"Disease classification accuracy: {test_acc:.3f}")
    print(f"Dynamic connectivity states: {n_states}")
    print(f"Features extracted shape: {results['features'].shape}")
    print("=" * 60)
    
    # Visualize results
    evaluator.visualize_results(results)


if __name__ == "__main__":
    main()
