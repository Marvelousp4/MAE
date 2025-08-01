"""
Simple training script for fMRI MAE

This script provides a minimal training loop for the fMRI MAE model
using spatiotemporal patch masking.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from fmri_mae import mae_fmri_base
from fmri_data_utils import generate_synthetic_fmri_data


def train_mae(model, data, epochs=50, batch_size=32, lr=1e-4, mask_ratio=0.75):
    """
    Train fMRI MAE model
    
    Args:
        model: MAE model
        data: Training data [N, P, T]
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        mask_ratio: Masking ratio
        
    Returns:
        Training history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    criterion = nn.MSELoss()
    
    history = {"loss": [], "recon_error": []}
    
    model.train()
    n_samples = data.size(0)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    print(f"Training on {device} with {n_samples} samples for {epochs} epochs")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    for epoch in range(epochs):
        epoch_losses = []
        epoch_errors = []
        
        # Shuffle data
        indices = torch.randperm(n_samples, device=device)
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            batch_data = data[batch_indices]
            
            # Forward pass
            optimizer.zero_grad()
            loss, pred, mask = model(batch_data, mask_ratio=mask_ratio)
            
            # Compute reconstruction error on visible patches for monitoring
            target = model.patchify(batch_data)
            visible_mask = mask.bool()
            if visible_mask.sum() > 0:
                recon_error = criterion(pred[visible_mask], target[visible_mask])
            else:
                recon_error = loss  # Fallback
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            epoch_errors.append(recon_error.item())
        
        avg_loss = np.mean(epoch_losses)
        avg_error = np.mean(epoch_errors)
        
        history["loss"].append(avg_loss)
        history["recon_error"].append(avg_error)
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.4f}, Recon Error = {avg_error:.4f}")
    
    return history


def visualize_training(history):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    ax1.plot(history["loss"], label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True)
    ax1.legend()
    
    # Reconstruction error
    ax2.plot(history["recon_error"], label="Reconstruction Error", color='orange')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE")
    ax2.set_title("Reconstruction Error")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def test_reconstruction(model, data, mask_ratio=0.75):
    """Test model reconstruction capability"""
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Take a single sample
        x = data[:1].to(device)
        
        # Forward pass
        loss, pred, mask = model(x, mask_ratio=mask_ratio)
        
        # Get original patches
        target = model.patchify(x)
        
        # Compute metrics
        mse = torch.mean((pred - target) ** 2).item()
        masked_mse = torch.mean(((pred - target) ** 2 * (1 - mask).unsqueeze(-1))).item()
        
        print(f"Reconstruction test:")
        print(f"  Overall MSE: {mse:.4f}")
        print(f"  Masked patches MSE: {masked_mse:.4f}")
        print(f"  Masking ratio: {(1 - mask.float().mean()).item():.2f}")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {pred.shape}")


def main():
    """Main training script"""
    print("=" * 60)
    print("fMRI MAE Training")
    print("=" * 60)
    
    # Generate synthetic data
    print("Generating synthetic fMRI data...")
    data = generate_synthetic_fmri_data(n_samples=500, n_regions=53, n_timepoints=200)
    print(f"Data shape: {data.shape}")
    
    # Create model
    print("Creating MAE model...")
    model = mae_fmri_base(
        num_regions=53,
        seq_len=200,
        patch_size_T=10
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    
    # Train model
    print("\nStarting training...")
    history = train_mae(
        model=model,
        data=data,
        epochs=50,
        batch_size=32,
        lr=1e-4,
        mask_ratio=0.75
    )
    
    # Test reconstruction
    print("\nTesting reconstruction...")
    test_reconstruction(model, data, mask_ratio=0.75)
    
    # Visualize results
    print("\nVisualizing training progress...")
    visualize_training(history)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
