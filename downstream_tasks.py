"""
Downstream tasks for fMRI MAE evaluation

This module implements various downstream tasks to evaluate the quality
of learned representations from the pretrained fMRI MAE model.

Tasks include:
1. FNC (Functional Network Connectivity) analysis
2. dFNC (dynamic Functional Network Connectivity) analysis  
3. Disease classification using pretrained features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class FNCAnalyzer:
    """
    Functional Network Connectivity (FNC) Analysis
    
    Analyzes connectivity patterns between brain networks using
    features extracted from pretrained MAE model.
    """
    
    def __init__(self, brain_networks: Dict[str, List[int]]):
        """
        Initialize FNC analyzer
        
        Args:
            brain_networks: Dictionary mapping network names to region indices
        """
        self.brain_networks = brain_networks
        self.network_names = list(brain_networks.keys())
        
    def extract_network_features(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features for each brain network
        
        Args:
            features: Extracted features [N, P, D] where P=53 regions
            
        Returns:
            Dictionary mapping network names to aggregated features
        """
        network_features = {}
        
        for network_name, regions in self.brain_networks.items():
            # Select features for this network's regions
            if len(regions) > 0:
                network_feat = features[:, regions, :]  # [N, num_regions, D]
                # Aggregate across regions (mean pooling)
                network_features[network_name] = network_feat.mean(dim=1)  # [N, D]
                
        return network_features
    
    def compute_fnc_matrix(self, network_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute FNC matrix between all network pairs using Pearson correlation
        
        Args:
            network_features: Features for each network
            
        Returns:
            FNC matrix [num_networks, num_networks]
        """
        num_networks = len(self.network_names)
        fnc_matrix = torch.zeros(num_networks, num_networks)
        
        for i, net1 in enumerate(self.network_names):
            for j, net2 in enumerate(self.network_names):
                if net1 in network_features and net2 in network_features:
                    feat1 = network_features[net1]  # [N, D]
                    feat2 = network_features[net2]  # [N, D]
                    
                    # Compute Pearson correlation between network features
                    # Average features across samples to get representative network vectors
                    mean_feat1 = feat1.mean(dim=0)  # [D]
                    mean_feat2 = feat2.mean(dim=0)  # [D]
                    
                    # Compute correlation coefficient
                    correlation = torch.corrcoef(torch.stack([mean_feat1, mean_feat2]))[0, 1]
                    
                    # Handle NaN values
                    if torch.isnan(correlation):
                        correlation = torch.tensor(0.0)
                        
                    fnc_matrix[i, j] = correlation
                    
        return fnc_matrix
    
    def visualize_fnc(self, fnc_matrix: torch.Tensor, title: str = "FNC Matrix"):
        """Visualize FNC matrix"""
        plt.figure(figsize=(10, 8))
        
        fnc_np = fnc_matrix.detach().cpu().numpy()
        sns.heatmap(fnc_np, 
                   xticklabels=self.network_names,
                   yticklabels=self.network_names,
                   annot=True, 
                   cmap='RdBu_r',
                   center=0,
                   square=True)
        
        plt.title(title)
        plt.tight_layout()
        plt.show()


class DynamicFNCAnalyzer:
    """
    Dynamic Functional Network Connectivity (dFNC) Analysis
    
    Analyzes time-varying connectivity patterns using sliding window approach.
    """
    
    def __init__(self, brain_networks: Dict[str, List[int]], window_size: int = 50, step_size: int = 10):
        """
        Initialize dFNC analyzer
        
        Args:
            brain_networks: Dictionary mapping network names to region indices
            window_size: Size of sliding window for temporal analysis
            step_size: Step size for sliding window
        """
        self.brain_networks = brain_networks
        self.network_names = list(brain_networks.keys())
        self.window_size = window_size
        self.step_size = step_size
        
    def extract_dynamic_features(self, time_series: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract dynamic features using sliding window
        
        Args:
            time_series: Time series data [N, P, T]
            
        Returns:
            List of feature tensors for each time window
        """
        N, P, T = time_series.shape
        dynamic_features = []
        
        # Sliding window extraction
        for start_t in range(0, T - self.window_size + 1, self.step_size):
            end_t = start_t + self.window_size
            window_data = time_series[:, :, start_t:end_t]  # [N, P, window_size]
            
            # Compute network connectivity for this window
            window_fnc = self._compute_window_fnc(window_data)
            dynamic_features.append(window_fnc)
            
        return dynamic_features
    
    def _compute_window_fnc(self, window_data: torch.Tensor) -> torch.Tensor:
        """Compute FNC matrix for a single time window"""
        N, P, T = window_data.shape
        num_networks = len(self.network_names)
        
        # Extract network time series
        network_ts = {}
        for network_name, regions in self.brain_networks.items():
            if len(regions) > 0:
                net_ts = window_data[:, regions, :].mean(dim=1)  # [N, T]
                network_ts[network_name] = net_ts
        
        # Compute correlation matrix
        fnc_matrices = []
        for n in range(N):
            sample_fnc = torch.zeros(num_networks, num_networks)
            
            for i, net1 in enumerate(self.network_names):
                for j, net2 in enumerate(self.network_names):
                    if net1 in network_ts and net2 in network_ts:
                        ts1 = network_ts[net1][n]  # [T]
                        ts2 = network_ts[net2][n]  # [T]
                        
                        # Compute Pearson correlation
                        corr = torch.corrcoef(torch.stack([ts1, ts2]))[0, 1]
                        if torch.isnan(corr):
                            corr = torch.tensor(0.0)
                        sample_fnc[i, j] = corr
                        
            fnc_matrices.append(sample_fnc)
            
        return torch.stack(fnc_matrices)  # [N, num_networks, num_networks]
    
    def cluster_dynamic_states(self, dynamic_features: List[torch.Tensor], n_states: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cluster dynamic connectivity states using k-means
        
        Args:
            dynamic_features: List of FNC matrices over time
            n_states: Number of connectivity states to identify
            
        Returns:
            state_labels: State labels for each time window
            state_centroids: Centroid FNC matrices for each state
        """
        # Flatten FNC matrices for clustering
        all_fnc = torch.cat(dynamic_features, dim=0)  # [N*num_windows, num_networks, num_networks]
        N_total, num_networks, _ = all_fnc.shape
        
        # Vectorize upper triangular part
        triu_indices = torch.triu_indices(num_networks, num_networks, offset=1)
        fnc_vectors = all_fnc[:, triu_indices[0], triu_indices[1]]  # [N_total, num_connections]
        
        # Simple k-means clustering (using scikit-learn would be better for real use)
        from sklearn.cluster import KMeans
        
        fnc_np = fnc_vectors.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=n_states, random_state=42)
        labels = kmeans.fit_predict(fnc_np)
        
        # Convert back to torch
        state_labels = torch.from_numpy(labels)
        
        # Compute state centroids
        state_centroids = []
        for state in range(n_states):
            state_mask = state_labels == state
            if state_mask.sum() > 0:
                state_fnc = all_fnc[state_mask].mean(dim=0)
                state_centroids.append(state_fnc)
        
        state_centroids = torch.stack(state_centroids) if state_centroids else torch.zeros(n_states, num_networks, num_networks)
        
        return state_labels, state_centroids


class DiseaseClassifier:
    """
    Disease classification using pretrained MAE features
    
    Uses extracted features from MAE encoder for downstream classification tasks.
    """
    
    def __init__(self, feature_dim: int, num_classes: int):
        """
        Initialize disease classifier
        
        Args:
            feature_dim: Dimension of input features
            num_classes: Number of disease classes
        """
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.classifier = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Prepare features for classification
        
        Args:
            features: Raw features [N, P, D] or [N, D]
            
        Returns:
            Processed features [N, feature_dim]
        """
        if features.dim() == 3:
            # Global average pooling across regions
            features = features.mean(dim=1)  # [N, D]
        
        return features
    
    def train_classifier(self, train_features: torch.Tensor, train_labels: torch.Tensor, 
                        val_features: Optional[torch.Tensor] = None, 
                        val_labels: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Train disease classifier
        
        Args:
            train_features: Training features [N_train, D]
            train_labels: Training labels [N_train]
            val_features: Validation features [N_val, D]
            val_labels: Validation labels [N_val]
            
        Returns:
            Training metrics
        """
        # Prepare features
        X_train = self.prepare_features(train_features)
        y_train = train_labels
        
        # Convert to numpy for sklearn
        X_train_np = X_train.detach().cpu().numpy()
        y_train_np = y_train.detach().cpu().numpy()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_np)
        
        # Train logistic regression classifier
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.classifier.fit(X_train_scaled, y_train_np)
        
        # Evaluate on training set
        train_pred = self.classifier.predict(X_train_scaled)
        train_acc = accuracy_score(y_train_np, train_pred)
        
        metrics = {"train_accuracy": train_acc}
        
        # Evaluate on validation set if provided
        if val_features is not None and val_labels is not None:
            X_val = self.prepare_features(val_features)
            X_val_np = X_val.detach().cpu().numpy()
            y_val_np = val_labels.detach().cpu().numpy()
            
            X_val_scaled = self.scaler.transform(X_val_np)
            val_pred = self.classifier.predict(X_val_scaled)
            val_acc = accuracy_score(y_val_np, val_pred)
            
            metrics["val_accuracy"] = val_acc
            
        return metrics
    
    def evaluate(self, test_features: torch.Tensor, test_labels: torch.Tensor) -> Dict[str, any]:
        """
        Evaluate classifier on test set
        
        Args:
            test_features: Test features [N_test, D]
            test_labels: Test labels [N_test]
            
        Returns:
            Evaluation metrics
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained yet!")
            
        # Prepare features
        X_test = self.prepare_features(test_features)
        X_test_np = X_test.detach().cpu().numpy()
        y_test_np = test_labels.detach().cpu().numpy()
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test_np)
        
        # Make predictions
        test_pred = self.classifier.predict(X_test_scaled)
        test_proba = self.classifier.predict_proba(X_test_scaled)
        
        # Compute metrics
        test_acc = accuracy_score(y_test_np, test_pred)
        class_report = classification_report(y_test_np, test_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test_np, test_pred)
        
        return {
            "test_accuracy": test_acc,
            "classification_report": class_report,
            "confusion_matrix": conf_matrix,
            "predictions": test_pred,
            "probabilities": test_proba
        }


def generate_synthetic_disease_data(n_samples: int = 200, n_regions: int = 53, n_timepoints: int = 200) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic fMRI data with disease labels for testing
    
    Args:
        n_samples: Number of samples
        n_regions: Number of brain regions
        n_timepoints: Number of timepoints
        
    Returns:
        fmri_data: Synthetic fMRI data [N, P, T]
        labels: Disease labels [N] (0=healthy, 1=disease)
    """
    # Generate labels (50% healthy, 50% disease)
    labels = torch.randint(0, 2, (n_samples,))
    
    # Generate base fMRI data
    fmri_data = torch.randn(n_samples, n_regions, n_timepoints)
    
    # Add disease-specific patterns
    for i in range(n_samples):
        if labels[i] == 1:  # Disease case
            # Increase connectivity in certain regions (e.g., default mode network)
            dmn_regions = [0, 5, 10, 15, 20]  # Example DMN regions
            fmri_data[i, dmn_regions, :] *= 1.5
            
            # Add more temporal variability
            fmri_data[i] += 0.3 * torch.randn_like(fmri_data[i])
    
    return fmri_data, labels


def test_downstream_tasks():
    """Test all downstream tasks"""
    print("=" * 60)
    print("Testing Downstream Tasks for fMRI MAE")
    print("=" * 60)
    
    # Define brain networks (simplified)
    brain_networks = {
        "DMN": [0, 5, 10, 15, 20],
        "Visual": [25, 30, 35, 40],
        "Sensorimotor": [1, 6, 11, 16],
        "Attention": [2, 7, 12, 17],
        "Frontal": [3, 8, 13, 18]
    }
    
    # Generate synthetic data
    n_samples = 100
    fmri_data, disease_labels = generate_synthetic_disease_data(n_samples=n_samples)
    print(f"Generated data shape: {fmri_data.shape}")
    print(f"Disease distribution: {disease_labels.bincount()}")
    
    # Simulate MAE features (normally these would come from trained MAE encoder)
    feature_dim = 256
    mae_features = torch.randn(n_samples, 53, feature_dim)  # [N, P, D]
    
    # Test 1: FNC Analysis
    print("\n" + "=" * 40)
    print("Test 1: FNC Analysis")
    print("=" * 40)
    
    fnc_analyzer = FNCAnalyzer(brain_networks)
    network_features = fnc_analyzer.extract_network_features(mae_features)
    
    print("Network features extracted:")
    for net_name, feat in network_features.items():
        print(f"  {net_name}: {feat.shape}")
    
    fnc_matrix = fnc_analyzer.compute_fnc_matrix(network_features)
    print(f"FNC matrix shape: {fnc_matrix.shape}")
    print("FNC matrix:")
    print(fnc_matrix.numpy())
    
    # Test 2: Dynamic FNC Analysis  
    print("\n" + "=" * 40)
    print("Test 2: Dynamic FNC Analysis")
    print("=" * 40)
    
    dfnc_analyzer = DynamicFNCAnalyzer(brain_networks, window_size=50, step_size=25)
    dynamic_features = dfnc_analyzer.extract_dynamic_features(fmri_data[:10])  # Use subset for speed
    
    print(f"Number of time windows: {len(dynamic_features)}")
    print(f"Each window FNC shape: {dynamic_features[0].shape}")
    
    # Cluster dynamic states
    state_labels, state_centroids = dfnc_analyzer.cluster_dynamic_states(dynamic_features, n_states=3)
    print(f"State labels shape: {state_labels.shape}")
    print(f"State centroids shape: {state_centroids.shape}")
    print(f"State distribution: {torch.bincount(state_labels)}")
    
    # Test 3: Disease Classification
    print("\n" + "=" * 40)
    print("Test 3: Disease Classification")
    print("=" * 40)
    
    # Split data
    train_idx, test_idx = train_test_split(range(n_samples), test_size=0.3, random_state=42)
    
    train_features = mae_features[train_idx]
    train_labels = disease_labels[train_idx]
    test_features = mae_features[test_idx]
    test_labels = disease_labels[test_idx]
    
    print(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
    
    # Train classifier
    classifier = DiseaseClassifier(feature_dim=feature_dim, num_classes=2)
    train_metrics = classifier.train_classifier(train_features, train_labels)
    
    print(f"Training accuracy: {train_metrics['train_accuracy']:.3f}")
    
    # Evaluate classifier
    test_metrics = classifier.evaluate(test_features, test_labels)
    print(f"Test accuracy: {test_metrics['test_accuracy']:.3f}")
    print("Classification report:")
    print(test_metrics['classification_report'])
    
    print("\n" + "=" * 60)
    print("All downstream tasks tested successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_downstream_tasks()
