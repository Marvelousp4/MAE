"""
fMRI Masked Autoencoder with Spatiotemporal Patch Masking

This implementation adapts MAE for fMRI data using spatiotemporal patch masking,
where each brain region's temporal sequence is divided into patches that serve
as tokens for fine-grained spatiotemporal modeling.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from functools import partial
from timm.models.vision_transformer import Block

# Import position embedding and masking
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.pos_embed import get_2d_sincos_pos_embed_new as get_2d_sincos_pos_embed
from src.models.fmri_masking import FMRISpatiotemporalMasking


class MaskedAutoencoderFMRI(nn.Module):
    """fMRI Masked Autoencoder with spatiotemporal patch masking"""

    def __init__(
        self,
        num_regions=53,
        seq_len=200,
        patch_size_T=10,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
    ):
        super().__init__()

        # Model configuration
        self.num_regions = num_regions
        self.seq_len = seq_len
        self.patch_size_T = patch_size_T
        
        assert seq_len % patch_size_T == 0, f"seq_len({seq_len}) must be divisible by patch_size_T({patch_size_T})"
        
        self.num_patches_T = seq_len // patch_size_T
        self.num_patches = num_regions * self.num_patches_T
        self.norm_pix_loss = norm_pix_loss

        # Masking strategy
        self.masker = FMRISpatiotemporalMasking()

        # Encoder
        self.patch_embed = nn.Linear(patch_size_T, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size_T, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize positional embeddings and model weights"""
        # 2D positional encoding for spatiotemporal patches
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            (self.num_regions, self.num_patches_T),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            (self.num_regions, self.num_patches_T),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize special tokens
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        
        # Initialize other weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights for linear layers and layer norms"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert fMRI data to spatiotemporal patches
        
        Args:
            x: Input data [N, P, T]
            
        Returns:
            Patches [N, num_patches, patch_size_T]
        """
        N, P, T = x.shape
        num_patches_T = T // self.patch_size_T
        
        # Reshape to patches: [N, P, T] -> [N, num_patches, patch_size_T]
        x = x.view(N, P, num_patches_T, self.patch_size_T)
        x = x.permute(0, 2, 1, 3).reshape(N, P * num_patches_T, self.patch_size_T)
        
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patches back to fMRI data
        
        Args:
            x: Patches [N, num_patches, patch_size_T]
            
        Returns:
            fMRI data [N, P, T]
        """
        N, num_patches, patch_size_T = x.shape
        P = self.num_regions
        num_patches_T = self.num_patches_T

        # [N, num_patches, patch_size_T] -> [N, P, T]
        x = x.view(N, num_patches_T, P, patch_size_T)
        x = x.permute(0, 2, 1, 3).reshape(N, P, num_patches_T * patch_size_T)
        
        return x

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float = 0.75) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder with masking
        
        Args:
            x: Input fMRI data [N, P, T]
            mask_ratio: Ratio of patches to mask
            
        Returns:
            latent: Encoded visible patches [N, L_visible+1, embed_dim]
            mask: Binary mask [N, num_patches]
            ids_restore: Indices for restoring patch order [N, num_patches]
        """
        # Apply spatiotemporal patch masking
        x_visible, mask, ids_keep, ids_restore = self.masker.spatiotemporal_patch_masking(
            x, self.patch_size_T, mask_ratio
        )

        # Embed visible patches
        x = self.patch_embed(x_visible)

        # Add positional embedding to visible patches
        pos_embed = self.pos_embed[:, 1:, :].repeat(x.shape[0], 1, 1)
        pos_embed_vis = torch.gather(
            pos_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[-1])
        )
        x = x + pos_embed_vis

        # Add class token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder
        
        Args:
            x: Encoded patches [N, L_visible+1, embed_dim]
            ids_restore: Indices for restoring patch order [N, num_patches]
            
        Returns:
            pred: Reconstructed patches [N, num_patches, patch_size_T]
        """
        # Embed tokens
        x = self.decoder_embed(x)

        # Add mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # Remove cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # Append cls token

        # Add positional embedding
        x = x + self.decoder_pos_embed

        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predictor projection
        x = self.decoder_pred(x)

        # Remove class token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, x: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss
        
        Args:
            x: Original fMRI data [N, P, T]
            pred: Predicted patches [N, num_patches, patch_size_T]
            mask: Binary mask [N, num_patches], 1 is keep, 0 is remove
            
        Returns:
            loss: Mean squared error loss on masked patches
        """
        target = self.patchify(x)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, num_patches], mean loss per patch

        # Compute loss only on masked patches
        loss = (loss * (1 - mask)).sum() / (1 - mask).sum()  # Mean loss on removed patches

        return loss

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.75) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass
        
        Args:
            x: Input fMRI data [N, P, T]
            mask_ratio: Ratio of patches to mask
            
        Returns:
            loss: Reconstruction loss
            pred: Predictions [N, num_patches, patch_size_T]
            mask: Binary mask [N, num_patches]
        """
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(x, pred, mask)
        
        return loss, pred, mask


# Model configurations
def mae_fmri_base(**kwargs):
    """Base fMRI MAE model (110M parameters)"""
    model = MaskedAutoencoderFMRI(
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def mae_fmri_large(**kwargs):
    """Large fMRI MAE model (307M parameters)"""
    model = MaskedAutoencoderFMRI(
        embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def mae_fmri_huge(**kwargs):
    """Huge fMRI MAE model (632M parameters)"""
    model = MaskedAutoencoderFMRI(
        embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
