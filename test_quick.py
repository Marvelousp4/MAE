"""
Quick test script for fMRI MAE
"""

import torch
from fmri_mae import mae_fmri_base
from fmri_data_utils import generate_synthetic_fmri_data


def quick_test():
    """Quick functionality test"""
    print("=" * 50)
    print("fMRI MAE Quick Test")
    print("=" * 50)
    
    # Generate small test data
    data = generate_synthetic_fmri_data(n_samples=16, n_regions=53, n_timepoints=200)
    print(f"Test data shape: {data.shape}")
    
    # Create model
    model = mae_fmri_base(num_regions=53, seq_len=200, patch_size_T=20)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        loss, pred, mask = model(data[:4], mask_ratio=0.75)
        
    print(f"Loss: {loss.item():.4f}")
    print(f"Prediction shape: {pred.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Masking ratio: {(1 - mask.float().mean()).item():.2f}")
    
    # Test different components
    print("\nTesting individual components:")
    
    # Test masking
    x_visible, mask, ids_keep, ids_restore = model.masker.spatiotemporal_patch_masking(
        data[:4], patch_size_T=20, mask_ratio=0.75
    )
    print(f"Visible patches shape: {x_visible.shape}")
    print(f"Number of patches: {mask.shape[1]}")
    print(f"Visible patches: {mask.sum(dim=1).tolist()}")
    
    # Test encoder
    latent, _, _ = model.forward_encoder(data[:4], mask_ratio=0.75)
    print(f"Encoded features shape: {latent.shape}")
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    quick_test()
