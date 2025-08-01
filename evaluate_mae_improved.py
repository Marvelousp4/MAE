"""
Improved End-to-end evaluation pipeline for fMRI MAE with model/data persistence

This script provides an improved pipeline with:
1. Model and data saving/loading for efficiency
2. Proper feature extraction from trained models
3. More realistic synthetic data generation
4. Better evaluation metrics
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
import os

from fmri_mae import MaskedAutoencoderFMRI
from fmri_masking import FMRISpatiotemporalMasking
from downstream_tasks import FNCAnalyzer, DynamicFNCAnalyzer, DiseaseClassifier, generate_synthetic_disease_data
from fmri_data_utils import load_or_generate_data, save_model, load_model


class ImprovedFMRIMAEEvaluator:
    """
    Improved evaluation pipeline for fMRI MAE with persistence
    """
    
    def __init__(self, model_config: Dict, brain_networks: Dict[str, List[int]], 
                 model_save_path: str = "pretrained_mae.pt"):
        """
        Initialize evaluator
        
        Args:
            model_config: MAE model configuration
            brain_networks: Brain network definitions
            model_save_path: Path to save/load the pretrained model
        """
        self.model_config = model_config
        self.brain_networks = brain_networks
        self.model_save_path = model_save_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize masking strategy
        self.masker = FMRISpatiotemporalMasking()
        
        # Initialize downstream analyzers
        self.fnc_analyzer = FNCAnalyzer(brain_networks)
        self.dfnc_analyzer = DynamicFNCAnalyzer(brain_networks)
        
        # Model will be loaded/created when needed
        self.mae_model = None
        
    def get_or_train_model(self, train_data: torch.Tensor, epochs: int = 50, 
                          force_retrain: bool = False) -> Dict[str, List[float]]:
        """
        Get pretrained model or train new one if needed
        
        Args:
            train_data: Training data [N, P, T]
            epochs: Number of training epochs if training is needed
            force_retrain: Force retraining even if model exists
            
        Returns:
            Training history (empty if loaded from file)
        """
        if not force_retrain and os.path.exists(self.model_save_path):
            print(f"Loading pretrained model from {self.model_save_path}")
            self.mae_model = load_model(self.model_save_path, MaskedAutoencoderFMRI)
            self.mae_model.to(self.device)
            return {"loss": [], "reconstruction_error": []}
        else:
            print("Training new MAE model...")
            # Initialize model
            self.mae_model = MaskedAutoencoderFMRI(**self.model_config)
            self.mae_model.to(self.device)
            
            # Train model
            history = self.pretrain_mae(train_data, epochs)
            
            # Save trained model
            save_model(self.mae_model, self.model_save_path)
            
            return history
    
    def pretrain_mae(self, train_data: torch.Tensor, epochs: int = 50, 
                     mask_ratio: float = 0.75) -> Dict[str, List[float]]:
        """
        Pretrain MAE model
        """
        print("Starting MAE pretraining...")
        
        optimizer = torch.optim.AdamW(self.mae_model.parameters(), lr=1e-4, weight_decay=0.05)
        criterion = nn.MSELoss()
        
        history = {"loss": [], "reconstruction_error": []}
        
        self.mae_model.train()
        train_data = train_data.to(self.device)
        
        for epoch in range(epochs):
            epoch_losses = []
            epoch_errors = []
            
            # Simple batch processing
            batch_size = min(32, train_data.size(0))
            n_batches = train_data.size(0) // batch_size
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, train_data.size(0))
                batch_data = train_data[start_idx:end_idx]
                
                # Forward pass
                optimizer.zero_grad()
                
                loss, pred, mask = self.mae_model(
                    batch_data, 
                    mask_ratio=mask_ratio
                )
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
                epoch_errors.append(loss.item())  # For MAE, loss is the reconstruction error
            
            avg_loss = np.mean(epoch_losses)
            avg_error = np.mean(epoch_errors)
            
            history["loss"].append(avg_loss)
            history["reconstruction_error"].append(avg_error)
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.4f}, Recon Error = {avg_error:.4f}")
        
        print("MAE pretraining completed!")
        return history
    
    def extract_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        Extract features using pretrained MAE encoder
        """
        if self.mae_model is None:
            raise ValueError("Model not loaded! Call get_or_train_model first.")
            
        self.mae_model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            # Use the encoder to extract features (no masking for feature extraction)
            features, _, _ = self.mae_model.forward_encoder(data, mask_ratio=0.0)
            
        return features.cpu()  # Return to CPU for downstream tasks
    
    def extract_region_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Convert patch features back to region-level features
        
        Args:
            features: Patch-level features [N, num_patches+1, embed_dim] (includes CLS token)
            
        Returns:
            Region-level features [N, num_regions, embed_dim]
        """
        N, total_patches, embed_dim = features.shape
        
        # Remove CLS token (first token)
        patch_features = features[:, 1:, :]  # [N, num_patches, embed_dim]
        
        num_regions = self.model_config.get('num_regions', 53)
        patches_per_region = patch_features.size(1) // num_regions
        
        # Reshape and aggregate patches to regions
        region_features = patch_features[:, :num_regions*patches_per_region, :]
        region_features = region_features.view(N, num_regions, patches_per_region, embed_dim)
        region_features = region_features.mean(dim=2)  # Average over patches within each region
        
        return region_features
    
    def evaluate_fnc(self, region_features: torch.Tensor) -> Dict[str, any]:
        """
        Evaluate FNC using region-level features
        """
        print("Evaluating FNC...")
        
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
        Evaluate dynamic FNC using original time series
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
    
    def evaluate_disease_classification(self, region_features: torch.Tensor, labels: torch.Tensor, 
                                      test_split: float = 0.3) -> Dict[str, any]:
        """
        Evaluate disease classification using region-level features
        """
        print("Evaluating disease classification...")
        
        # Global average pooling for classification
        N, num_regions, embed_dim = region_features.shape
        pooled_features = region_features.mean(dim=1)  # [N, embed_dim]
        
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
    
    def run_complete_evaluation(self, train_data: torch.Tensor, eval_data: torch.Tensor, labels: torch.Tensor, 
                               pretrain_epochs: int = 50, force_retrain: bool = False) -> Dict[str, any]:
        """
        Run complete evaluation pipeline with model persistence
        """
        print("=" * 60)
        print("Running Improved fMRI MAE Evaluation Pipeline")
        print("=" * 60)
        
        results = {}
        
        # Step 1: Get or train MAE model
        print("\n" + "=" * 40)
        print("Step 1: MAE Model Preparation")
        print("=" * 40)
        
        pretrain_history = self.get_or_train_model(train_data, epochs=pretrain_epochs, force_retrain=force_retrain)
        results["pretrain_history"] = pretrain_history
        
        # Step 2: Extract features from evaluation data
        print("\n" + "=" * 40)
        print("Step 2: Feature Extraction")
        print("=" * 40)
        
        patch_features = self.extract_features(eval_data)
        print(f"Patch features shape: {patch_features.shape}")
        
        region_features = self.extract_region_features(patch_features)
        print(f"Region features shape: {region_features.shape}")
        
        results["patch_features"] = patch_features
        results["region_features"] = region_features
        
        # Step 3: FNC evaluation
        print("\n" + "=" * 40)
        print("Step 3: FNC Evaluation")
        print("=" * 40)
        
        fnc_results = self.evaluate_fnc(region_features)
        print(f"FNC matrix shape: {fnc_results['fnc_matrix'].shape}")
        results["fnc"] = fnc_results
        
        # Step 4: dFNC evaluation
        print("\n" + "=" * 40)
        print("Step 4: Dynamic FNC Evaluation")
        print("=" * 40)
        
        dfnc_results = self.evaluate_dfnc(eval_data)
        print(f"Number of time windows: {dfnc_results['num_windows']}")
        print(f"Dynamic states shape: {dfnc_results['state_centroids'].shape}")
        results["dfnc"] = dfnc_results
        
        # Step 5: Disease classification
        print("\n" + "=" * 40)
        print("Step 5: Disease Classification")
        print("=" * 40)
        
        classification_results = self.evaluate_disease_classification(region_features, labels)
        print(f"Test accuracy: {classification_results['test_metrics']['test_accuracy']:.3f}")
        results["classification"] = classification_results
        
        print("\n" + "=" * 60)
        print("Complete evaluation pipeline finished!")
        print("=" * 60)
        
        return results
    
    def visualize_results(self, results: Dict[str, any]):
        """Visualize evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Training loss (if available)
        ax1 = axes[0, 0]
        if results["pretrain_history"]["loss"]:
            ax1.plot(results["pretrain_history"]["loss"], label="Training Loss", linewidth=2)
            ax1.plot(results["pretrain_history"]["reconstruction_error"], label="Reconstruction Error", linewidth=2)
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("MAE Training Progress")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "Model loaded from file\n(No training history)", 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title("MAE Model Status")
        
        # Plot 2: FNC matrix
        ax2 = axes[0, 1]
        fnc_matrix = results["fnc"]["fnc_matrix"].detach().cpu().numpy()
        im = ax2.imshow(fnc_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax2.set_title("Functional Network Connectivity")
        ax2.set_xlabel("Network")
        ax2.set_ylabel("Network")
        plt.colorbar(im, ax=ax2)
        
        # Plot 3: Dynamic states
        ax3 = axes[1, 0]
        state_labels = results["dfnc"]["state_labels"].cpu().numpy()
        ax3.hist(state_labels, bins=len(np.unique(state_labels)), alpha=0.7, edgecolor='black')
        ax3.set_xlabel("Dynamic State")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Dynamic Connectivity States")
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Feature distribution (region features)
        ax4 = axes[1, 1]
        region_features = results["region_features"].cpu().numpy()
        ax4.hist(region_features.flatten(), bins=50, alpha=0.7, edgecolor='black')
        ax4.set_xlabel("Feature Value")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Extracted Region Feature Distribution")
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("evaluation_results.png")
        plt.show()


def main():
    """Main evaluation script"""
    print("Improved fMRI MAE Evaluation Pipeline")
    print("=" * 60)
    
    # Define brain networks (more realistic)
    brain_networks = {
        "DMN": [0, 5, 10, 15, 20, 25, 30],
        "Visual": [35, 40, 45, 50],
        "Sensorimotor": [1, 6, 11, 16, 21, 26],
        "Attention": [2, 7, 12, 17, 22],
        "Frontal": [3, 8, 13, 18, 23],
        "Parietal": [4, 9, 14, 19, 24]
    }
    
    # Model configuration
    model_config = {
        "num_regions": 53,
        "seq_len": 200,
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
    
    # Load or generate data
    print("Loading or generating fMRI data...")
    fmri_data = load_or_generate_data("fmri_training_data.pt", n_samples=500, n_regions=53, n_timepoints=200)
    
    # Generate disease labels
    print("Generating disease data for evaluation...")
    eval_data, disease_labels = generate_synthetic_disease_data(
        n_samples=200, n_regions=53, n_timepoints=200
    )
    
    print(f"Training data shape: {fmri_data.shape}")
    print(f"Evaluation data shape: {eval_data.shape}")
    print(f"Label distribution: {disease_labels.bincount()}")
    
    # Initialize evaluator
    evaluator = ImprovedFMRIMAEEvaluator(model_config, brain_networks, "pretrained_mae_model.pt")
    
    # Run complete evaluation
    results = evaluator.run_complete_evaluation(
        fmri_data, eval_data, disease_labels, pretrain_epochs=50, force_retrain=False
    )
    
    # Visualize results
    print("\nGenerating visualization...")
    evaluator.visualize_results(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    if results['pretrain_history']['loss']:
        print(f"Final training loss: {results['pretrain_history']['loss'][-1]:.4f}")
    else:
        print("Model loaded from saved file")
    print(f"Test classification accuracy: {results['classification']['test_metrics']['test_accuracy']:.3f}")
    print(f"Number of dynamic states identified: {len(torch.unique(results['dfnc']['state_labels']))}")
    print(f"Region features mean: {results['region_features'].mean():.4f}")
    print(f"Region features std: {results['region_features'].std():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
