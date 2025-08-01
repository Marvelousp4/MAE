#!/usr/bin/env python3
"""
fMRI data generation and loading utilities
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os


class FMRIDataset(Dataset):
    """fMRI dataset for MAE training"""

    def __init__(
        self,
        data_path=None,
        num_subjects=50,
        num_regions=53,
        seq_len=200,
        generate_synthetic=True,
    ):
        """
        Initialize fMRI dataset

        Args:
            data_path: Path to real fMRI data directory
            num_subjects: Number of subjects for synthetic data
            num_regions: Number of brain regions
            seq_len: Sequence length (timepoints)
            generate_synthetic: Whether to generate synthetic data
        """
        self.num_regions = num_regions
        self.seq_len = seq_len

        if data_path and os.path.exists(data_path):
            self.data = self.load_real_data(data_path)
        elif generate_synthetic:
            self.data = self.generate_synthetic_data(num_subjects, num_regions, seq_len)
        else:
            raise ValueError(
                "Either provide valid data_path or set generate_synthetic=True"
            )

    def load_real_data(self, data_path):
        """Load real fMRI data"""
        data_files = [f for f in os.listdir(data_path) if f.endswith(".npy")]
        data_list = []

        for file in data_files:
            subject_data = np.load(os.path.join(data_path, file))
            if subject_data.shape == (self.num_regions, self.seq_len):
                data_list.append(subject_data)

        return (
            np.stack(data_list)
            if data_list
            else self.generate_synthetic_data(50, self.num_regions, self.seq_len)
        )

    def generate_synthetic_data(self, num_subjects, num_regions, seq_len):
        """Generate more realistic synthetic fMRI data with network patterns and noise"""
        np.random.seed(42)  # For reproducibility
        
        # Initialize data with baseline noise
        data = np.random.randn(num_subjects, num_regions, seq_len) * 0.3
        
        # Define time vector
        t = np.linspace(0, seq_len * 0.72, seq_len)  # Assuming TR=0.72s
        
        # Add realistic brain network patterns with subject variability
        for subj in range(num_subjects):
            subject_noise_scale = 0.8 + 0.4 * np.random.random()  # Individual differences
            
            # Default Mode Network (DMN) - low frequency oscillations
            dmn_regions = [0, 1, 2, 5, 8, 12, 15, 18, 22, 26, 30]
            dmn_signal = (0.4 * np.sin(0.08 * t) + 0.3 * np.sin(0.04 * t) + 
                         0.2 * np.sin(0.12 * t)) * subject_noise_scale
            for region in dmn_regions:
                if region < num_regions:
                    # Add phase differences between regions
                    phase_shift = np.random.uniform(0, 0.5)
                    data[subj, region] += dmn_signal * (0.8 + 0.4 * np.random.random()) + \
                                        0.1 * np.sin(0.08 * t + phase_shift)

            # Visual Network - intermediate frequency
            visual_regions = [35, 40, 45, 50, 52]
            visual_signal = (0.35 * np.sin(0.1 * t) + 0.25 * np.sin(0.2 * t)) * subject_noise_scale
            for region in visual_regions:
                if region < num_regions:
                    phase_shift = np.random.uniform(0, 0.3)
                    data[subj, region] += visual_signal * (0.7 + 0.6 * np.random.random()) + \
                                        0.1 * np.sin(0.1 * t + phase_shift)

            # Sensorimotor Network - higher frequency
            smn_regions = [6, 9, 13, 17, 20, 24, 28, 32]
            smn_signal = (0.3 * np.sin(0.15 * t) + 0.2 * np.sin(0.08 * t)) * subject_noise_scale
            for region in smn_regions:
                if region < num_regions:
                    phase_shift = np.random.uniform(0, 0.4)
                    data[subj, region] += smn_signal * (0.6 + 0.8 * np.random.random()) + \
                                        0.1 * np.sin(0.15 * t + phase_shift)

            # Attention Network
            attention_regions = [2, 7, 12, 17, 22, 27, 33, 38]
            attention_signal = (0.25 * np.sin(0.12 * t) + 0.2 * np.sin(0.06 * t)) * subject_noise_scale
            for region in attention_regions:
                if region < num_regions:
                    phase_shift = np.random.uniform(0, 0.6)
                    data[subj, region] += attention_signal * (0.5 + 0.7 * np.random.random()) + \
                                        0.1 * np.sin(0.12 * t + phase_shift)

            # Executive Control Network
            executive_regions = [3, 8, 13, 18, 23, 29, 34, 39, 44, 49]
            executive_signal = (0.28 * np.sin(0.1 * t) + 0.18 * np.sin(0.05 * t)) * subject_noise_scale
            for region in executive_regions:
                if region < num_regions:
                    phase_shift = np.random.uniform(0, 0.5)
                    data[subj, region] += executive_signal * (0.6 + 0.6 * np.random.random()) + \
                                        0.1 * np.sin(0.1 * t + phase_shift)
            
            # Add some global signal (realistic in fMRI)
            global_signal = 0.1 * np.sin(0.02 * t) * subject_noise_scale
            data[subj, :, :] += global_signal
            
            # Add region-specific noise
            for region in range(num_regions):
                region_noise = np.random.randn(seq_len) * 0.15
                data[subj, region] += region_noise

        return data

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])


def create_fmri_dataloader(
    data_path=None, batch_size=32, num_workers=4, **dataset_kwargs
):
    """Create fMRI data loader"""
    dataset = FMRIDataset(data_path=data_path, **dataset_kwargs)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader


def save_synthetic_data(save_dir, num_subjects=50, num_regions=53, seq_len=200):
    """Save synthetic fMRI data to disk"""
    os.makedirs(save_dir, exist_ok=True)

    dataset = FMRIDataset(
        generate_synthetic=True,
        num_subjects=num_subjects,
        num_regions=num_regions,
        seq_len=seq_len,
    )

    # Save individual subject files
    for i, data in enumerate(dataset.data):
        np.save(os.path.join(save_dir, f"subject_{i:03d}.npy"), data)

    # Save all subjects in one file
    np.save(os.path.join(save_dir, "all_subjects.npy"), dataset.data)

    print(f"Saved {num_subjects} synthetic fMRI datasets to {save_dir}")
    print(f"Data shape per subject: {dataset.data.shape[1:]}")


def generate_synthetic_fmri_data(n_samples=100, n_regions=53, n_timepoints=200):
    """
    Generate synthetic fMRI data for testing
    
    Args:
        n_samples: Number of samples
        n_regions: Number of brain regions  
        n_timepoints: Number of timepoints
        
    Returns:
        Synthetic fMRI data [N, P, T]
    """
    dataset = FMRIDataset(
        generate_synthetic=True,
        num_subjects=n_samples,
        num_regions=n_regions,
        seq_len=n_timepoints,
    )
    return torch.FloatTensor(dataset.data)


def load_or_generate_data(data_file="fmri_data.pt", n_samples=500, n_regions=53, n_timepoints=200):
    """Load existing data or generate new synthetic data"""
    if os.path.exists(data_file):
        print(f"Loading existing data from {data_file}")
        return torch.load(data_file)
    else:
        print(f"Generating new synthetic data and saving to {data_file}")
        data = generate_synthetic_fmri_data(n_samples, n_regions, n_timepoints)
        torch.save(data, data_file)
        return data


def save_model(model, filepath):
    """Save trained model"""
    # Get model configuration from constructor parameters
    model_config = {
        'num_regions': model.num_regions,
        'seq_len': model.seq_len,
        'patch_size_T': model.patch_size_T,
        'embed_dim': model.patch_embed.out_features,  # Get from layer
        'depth': len(model.blocks),
        'num_heads': model.blocks[0].attn.num_heads,
        'decoder_embed_dim': model.decoder_embed.out_features,
        'decoder_depth': len(model.decoder_blocks),
        'decoder_num_heads': model.decoder_blocks[0].attn.num_heads,
        'mlp_ratio': 4.0,  # Default value
        'norm_pix_loss': model.norm_pix_loss,
    }
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath, model_class):
    """Load trained model"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model_config = checkpoint['model_config']
    
    # Create model with saved config
    model = model_class(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {filepath}")
    return model


if __name__ == "__main__":
    # Generate and save synthetic data
    save_synthetic_data("fmri_data", num_subjects=50)

    # Test data loading
    dataloader = create_fmri_dataloader(data_path="fmri_data", batch_size=8)

    for batch in dataloader:
        print(f"Batch shape: {batch.shape}")
        break
