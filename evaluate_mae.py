"""
End-to-end evaluation pipeline for fMRI MAE

This script provides a complete pipeline for:
1. Pretraining MAE on fMRI data
2. Extracting features using pretrained encoder
3. Evaluating on downstream tasks (FNC, dFNC, disease classification)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns

from fmri_mae import MaskedAutoencoderFMRI
from fmri_masking import FMRISpatiotemporalMasking
from downstream_tasks import FNCAnalyzer, DynamicFNCAnalyzer, DiseaseClassifier, generate_synthetic_disease_data


class FMRIMAEEvaluator:
    """
    Complete evaluation pipeline for fMRI MAE
    """
    
    def __init__(self, model_config: Dict, brain_networks: Dict[str, List[int]]):
        """
        Initialize evaluator
        
        Args:
            model_config: MAE model configuration
            brain_networks: Brain network definitions
        """
        self.model_config = model_config
        self.brain_networks = brain_networks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.mae_model = MaskedAutoencoderFMRI(**model_config)
        self.mae_model.to(self.device)
        
        # Initialize masking strategy
        self.masker = FMRISpatiotemporalMasking()
        
        # Initialize downstream analyzers
        self.fnc_analyzer = FNCAnalyzer(brain_networks)
        self.dfnc_analyzer = DynamicFNCAnalyzer(brain_networks)
        
    def pretrain_mae(self, train_data: torch.Tensor, epochs: int = 10, 
                     patch_size_T: int = 20, mask_ratio: float = 0.75) -> Dict[str, List[float]]:
        """
        Pretrain MAE model
        
        Args:
            train_data: Training data [N, P, T]
            epochs: Number of training epochs
            patch_size_T: Temporal patch size
            mask_ratio: Masking ratio
            
        Returns:
            Training history
        """
        print("Starting MAE pretraining...")
        
        optimizer = torch.optim.AdamW(self.mae_model.parameters(), lr=1e-4, weight_decay=0.05)
        criterion = nn.MSELoss()
        
        history = {"loss": [], "reconstruction_error": []}
        
        self.mae_model.train()
        
        for epoch in range(epochs):
            epoch_losses = []
            epoch_errors = []
            
            # Simple batch processing (for real use, implement proper DataLoader)
            batch_size = min(32, train_data.size(0))
            n_batches = train_data.size(0) // batch_size
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, train_data.size(0))
                batch_data = train_data[start_idx:end_idx].to(self.device)
                
                # Apply masking
                x_visible, mask, ids_keep, ids_restore = self.masker.spatiotemporal_patch_masking(
                    batch_data, patch_size_T=patch_size_T, mask_ratio=mask_ratio
                )
                
                # Forward pass
                optimizer.zero_grad()
                
                loss, pred, mask = self.mae_model(
                    batch_data, 
                    mask_ratio=mask_ratio
                )
                
                # Compute reconstruction error (same as loss for MAE)
                recon_error = loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
                epoch_errors.append(recon_error.item())
            
            avg_loss = np.mean(epoch_losses)
            avg_error = np.mean(epoch_errors)
            
            history["loss"].append(avg_loss)
            history["reconstruction_error"].append(avg_error)
            
            if epoch % 2 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, Recon Error = {avg_error:.4f}")
        
        print("MAE pretraining completed!")
        return history
    
    def extract_features(self, data: torch.Tensor, patch_size_T: int = 20) -> torch.Tensor:
        """
        Extract features using pretrained MAE encoder
        
        Args:
            data: Input data [N, P, T]
            patch_size_T: Temporal patch size used during pretraining
            
        Returns:
            Extracted features [N, num_patches, embed_dim]
        """
        self.mae_model.eval()
        data = data.to(self.device)  # Move data to the correct device
        
        with torch.no_grad():
            # Apply masking to get visible patches only
            x_visible, mask, ids_keep, ids_restore = self.masker.spatiotemporal_patch_masking(
                data, patch_size_T=patch_size_T, mask_ratio=0.0  # No masking for feature extraction
            )
            
            # Encode visible patches using the trained encoder
            features, _, _ = self.mae_model.forward_encoder(data, mask_ratio=0.0)
            
        return features
    
    def evaluate_fnc(self, features: torch.Tensor) -> Dict[str, any]:
        """
        Evaluate FNC using extracted features
        
        Args:
            features: Extracted features [N, num_patches, embed_dim]
            
        Returns:
            FNC evaluation results
        """
        print("Evaluating FNC...")
        
        # Convert patch features back to region features
        # For simplicity, we'll aggregate patches back to regions
        N, num_patches, embed_dim = features.shape
        P = 53  # Number of brain regions
        
        # Reshape features to approximate region-level features
        patches_per_region = num_patches // P
        region_features = features[:, :P*patches_per_region, :].view(N, P, patches_per_region, embed_dim)
        region_features = region_features.mean(dim=2)  # [N, P, embed_dim]
        
        # Extract network features
        network_features = self.fnc_analyzer.extract_network_features(region_features)
        
        # Compute FNC matrix
        fnc_matrix = self.fnc_analyzer.compute_fnc_matrix(network_features)
        
        return {
            "network_features": network_features,
            "fnc_matrix": fnc_matrix,
            "region_features": region_features
        }
    
    def evaluate_dfnc(self, data: torch.Tensor) -> Dict[str, any]:
        """
        Evaluate dynamic FNC
        
        Args:
            data: Original time series data [N, P, T]
            
        Returns:
            dFNC evaluation results
        """
        print("Evaluating dynamic FNC...")
        
        # Extract dynamic features
        dynamic_features = self.dfnc_analyzer.extract_dynamic_features(data)
        
        # Cluster dynamic states
        state_labels, state_centroids = self.dfnc_analyzer.cluster_dynamic_states(
            dynamic_features, n_states=4
        )
        
        return {
            "dynamic_features": dynamic_features,
            "state_labels": state_labels,
            "state_centroids": state_centroids,
            "num_windows": len(dynamic_features)
        }
    
    def evaluate_disease_classification(self, features: torch.Tensor, labels: torch.Tensor, 
                                      test_split: float = 0.3) -> Dict[str, any]:
        """
        Evaluate disease classification using extracted features
        
        Args:
            features: Extracted features [N, num_patches, embed_dim]
            labels: Disease labels [N]
            test_split: Fraction of data to use for testing
            
        Returns:
            Classification evaluation results
        """
        print("Evaluating disease classification...")
        
        # Prepare features for classification (global average pooling)
        N, num_patches, embed_dim = features.shape
        pooled_features = features.mean(dim=1)  # [N, embed_dim]
        
        # Split data
        n_test = int(N * test_split)
        indices = torch.randperm(N)
        
        train_indices = indices[n_test:]
        test_indices = indices[:n_test]
        
        train_features = pooled_features[train_indices]
        train_labels = labels[train_indices]
        test_features = pooled_features[test_indices]
        test_labels = labels[test_indices]
        
        # Train classifier
        classifier = DiseaseClassifier(feature_dim=embed_dim, num_classes=2)
        train_metrics = classifier.train_classifier(train_features, train_labels)
        
        # Evaluate classifier
        test_metrics = classifier.evaluate(test_features, test_labels)
        
        return {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "classifier": classifier
        }
    
    def run_complete_evaluation(self, data: torch.Tensor, labels: torch.Tensor, 
                               pretrain_epochs: int = 10) -> Dict[str, any]:
        """
        Run complete evaluation pipeline
        
        Args:
            data: fMRI data [N, P, T]
            labels: Disease labels [N]
            pretrain_epochs: Number of pretraining epochs
            
        Returns:
            Complete evaluation results
        """
        print("=" * 60)
        print("Running Complete fMRI MAE Evaluation Pipeline")
        print("=" * 60)
        
        results = {}
        
        # Step 1: Pretrain MAE
        print("\n" + "=" * 40)
        print("Step 1: MAE Pretraining")
        print("=" * 40)
        
        pretrain_history = self.pretrain_mae(data, epochs=pretrain_epochs)
        results["pretrain_history"] = pretrain_history
        
        # Step 2: Extract features
        print("\n" + "=" * 40)
        print("Step 2: Feature Extraction")
        print("=" * 40)
        
        features = self.extract_features(data)
        print(f"Extracted features shape: {features.shape}")
        results["features"] = features
        
        # Step 3: FNC evaluation
        print("\n" + "=" * 40)
        print("Step 3: FNC Evaluation")
        print("=" * 40)
        
        fnc_results = self.evaluate_fnc(features)
        print(f"FNC matrix shape: {fnc_results['fnc_matrix'].shape}")
        results["fnc"] = fnc_results
        
        # Step 4: dFNC evaluation
        print("\n" + "=" * 40)
        print("Step 4: Dynamic FNC Evaluation")
        print("=" * 40)
        
        dfnc_results = self.evaluate_dfnc(data)
        print(f"Number of time windows: {dfnc_results['num_windows']}")
        print(f"Dynamic states shape: {dfnc_results['state_centroids'].shape}")
        results["dfnc"] = dfnc_results
        
        # Step 5: Disease classification
        print("\n" + "=" * 40)
        print("Step 5: Disease Classification")
        print("=" * 40)
        
        classification_results = self.evaluate_disease_classification(features, labels)
        print(f"Test accuracy: {classification_results['test_metrics']['test_accuracy']:.3f}")
        results["classification"] = classification_results
        
        print("\n" + "=" * 60)
        print("Complete evaluation pipeline finished!")
        print("=" * 60)
        
        return results
    
    def visualize_results(self, results: Dict[str, any]):
        """Visualize evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Training loss
        ax1 = axes[0, 0]
        ax1.plot(results["pretrain_history"]["loss"], label="Training Loss")
        ax1.plot(results["pretrain_history"]["reconstruction_error"], label="Reconstruction Error")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("MAE Training Progress")
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: FNC matrix
        ax2 = axes[0, 1]
        fnc_matrix = results["fnc"]["fnc_matrix"].detach().cpu().numpy()
        im = ax2.imshow(fnc_matrix, cmap='RdBu_r', aspect='auto')
        ax2.set_title("Functional Network Connectivity")
        ax2.set_xlabel("Network")
        ax2.set_ylabel("Network")
        plt.colorbar(im, ax=ax2)
        
        # Plot 3: Dynamic states
        ax3 = axes[1, 0]
        state_labels = results["dfnc"]["state_labels"].cpu().numpy()
        ax3.hist(state_labels, bins=len(np.unique(state_labels)), alpha=0.7)
        ax3.set_xlabel("Dynamic State")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Dynamic Connectivity States")
        ax3.grid(True)
        
        # Plot 4: Classification performance
        ax4 = axes[1, 1]
        conf_matrix = results["classification"]["test_metrics"]["confusion_matrix"]
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax4)
        ax4.set_xlabel("Predicted")
        ax4.set_ylabel("Actual")
        ax4.set_title("Disease Classification Confusion Matrix")
        
        plt.tight_layout()
        plt.show()


def main():
    """Main evaluation script"""
    print("fMRI MAE Complete Evaluation Pipeline")
    print("=" * 60)
    
    # Define brain networks
    brain_networks = {
        "DMN": [0, 5, 10, 15, 20, 25],
        "Visual": [30, 35, 40, 45],
        "Sensorimotor": [1, 6, 11, 16, 21],
        "Attention": [2, 7, 12, 17, 22],
        "Frontal": [3, 8, 13, 18, 23],
        "Parietal": [4, 9, 14, 19, 24]
    }
    
    # Model configuration
    model_config = {
        "patch_size_T": 10,
        "embed_dim": 256,
        "depth": 6,
        "num_heads": 8,
        "decoder_embed_dim": 128,
        "decoder_depth": 4,
        "decoder_num_heads": 4,
        "mlp_ratio": 4.0,
        "norm_layer": nn.LayerNorm
    }
    
    # Generate synthetic data
    print("Generating synthetic fMRI data...")
    n_samples = 200
    fmri_data, disease_labels = generate_synthetic_disease_data(
        n_samples=n_samples, n_regions=53, n_timepoints=200
    )
    
    print(f"Data shape: {fmri_data.shape}")
    print(f"Label distribution: {disease_labels.bincount()}")
    
    # Initialize evaluator
    evaluator = FMRIMAEEvaluator(model_config, brain_networks)
    
    # Run complete evaluation
    results = evaluator.run_complete_evaluation(
        fmri_data, disease_labels, pretrain_epochs=5
    )
    
    # Visualize results
    print("\nGenerating visualization...")
    evaluator.visualize_results(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Final training loss: {results['pretrain_history']['loss'][-1]:.4f}")
    print(f"Final reconstruction error: {results['pretrain_history']['reconstruction_error'][-1]:.4f}")
    print(f"Test classification accuracy: {results['classification']['test_metrics']['test_accuracy']:.3f}")
    print(f"Number of dynamic states identified: {len(torch.unique(results['dfnc']['state_labels']))}")
    print("=" * 60)


if __name__ == "__main__":
    main()
