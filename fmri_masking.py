#!/usr/bin/env python3
"""
Spatiotemporal patch masking for fMRI Masked Autoencoder (MAE)
"""

import torch
import numpy as np
from typing import Tuple


class FMRISpatiotemporalMasking:
    """Spatiotemporal patch masking for fMRI MAE"""

    def __init__(self):
        pass

    def spatiotemporal_patch_masking(
        self, x: torch.Tensor, patch_size_T: int, mask_ratio: float = 0.75
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Spatiotemporal patch masking for fine-grained brain dynamics modeling
        
        Divides each brain region's temporal sequence into patches.
        Each patch becomes a token for fine-grained spatiotemporal modeling.

        Args:
            x: Input data [N, P, T]
            patch_size_T: Temporal patch size  
            mask_ratio: Masking ratio

        Returns:
            x_visible: Visible patches [N, L_visible, patch_size_T]
            mask: Binary mask [N, num_patches], 1=visible, 0=masked
            ids_keep: Indices of kept patches [N, L_visible]
            ids_restore: Indices to restore original order [N, num_patches]
        """
        N, P, T = x.shape
        if T % patch_size_T != 0:
            raise ValueError(
                f"T({T}) must be divisible by patch_size_T({patch_size_T})"
            )

        num_patches_T = T // patch_size_T
        num_patches = P * num_patches_T

        # Reshape to patches: [N, P, T] -> [N, num_patches, patch_size_T]
        x_patched = x.view(N, P, num_patches_T, patch_size_T)
        x_patched = x_patched.permute(0, 2, 1, 3).reshape(N, num_patches, patch_size_T)

        # Random masking
        L_visible = int(num_patches * (1 - mask_ratio))

        noise = torch.rand(N, num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :L_visible]
        x_visible = torch.gather(
            x_patched, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, patch_size_T)
        )

        mask = torch.zeros([N, num_patches], device=x.device)
        mask[:, :L_visible] = 1
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_visible, mask, ids_keep, ids_restore


def test_spatiotemporal_masking():
    """Test spatiotemporal patch masking"""
    print("=" * 60)
    print("Testing Spatiotemporal Patch Masking")
    print("=" * 60)

    # Create test data
    N, P, T = 4, 53, 200  # 4 samples, 53 regions, 200 timepoints
    x = torch.randn(N, P, T)
    print(f"Input data shape: {x.shape}")

    masker = FMRISpatiotemporalMasking()

    patch_size_T = 20
    x_visible_patch, patch_mask, ids_keep_patch, ids_restore_patch = (
        masker.spatiotemporal_patch_masking(
            x, patch_size_T=patch_size_T, mask_ratio=0.75
        )
    )
    
    print(f"x_visible_patch shape: {x_visible_patch.shape}")
    print(f"patch_mask shape: {patch_mask.shape}")
    print(f"Patch size: {patch_size_T}")
    print(f"Total patches: {P * (T // patch_size_T)}")
    print(f"Visible patches: {patch_mask.sum(dim=1)}")
    print(f"Masked patches: {P * (T // patch_size_T) - patch_mask.sum(dim=1)}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_spatiotemporal_masking()
