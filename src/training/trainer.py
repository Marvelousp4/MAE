"""
Scientific-grade MAE training module with proper validation and monitoring.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
import os
from pathlib import Path

from ..models.fmri_mae import MaskedAutoencoderFMRI


class MAETrainer:
    """
    Scientific-grade MAE trainer with proper validation monitoring and early stopping.
    """
    
    def __init__(self, model: MaskedAutoencoderFMRI, config: Dict[str, Any]):
        """
        Initialize trainer.
        
        Args:
            model: MAE model instance
            config: Training configuration dictionary
        """
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.05)
        )
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Training state
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }
        
        # Early stopping
        self.patience = config.get('patience', 20)
        self.patience_counter = 0
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for training monitoring."""
        log_dir = Path(self.config.get('log_dir', 'outputs/logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger = logging.getLogger('MAETrainer')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_dir / 'training.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def train(self, train_data: torch.Tensor, val_data: torch.Tensor, 
              epochs: int = 100) -> Dict[str, Any]:
        """
        Train MAE with proper validation monitoring.
        
        Args:
            train_data: Training data [N, P, T]
            val_data: Validation data [N_val, P, T]
            epochs: Number of training epochs
            
        Returns:
            Training history dictionary
        """
        self.logger.info(f"Starting MAE training for {epochs} epochs")
        self.logger.info(f"Train data: {train_data.shape}, Val data: {val_data.shape}")
        self.logger.info(f"Device: {self.device}")
        
        # Create data loaders
        train_loader = self._create_dataloader(train_data, shuffle=True)
        val_loader = self._create_dataloader(val_data, shuffle=False)
        
        # Training loop
        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader, epoch)
            val_loss = self._validate_epoch(val_loader, epoch)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['epochs'].append(epoch)
            
            # Check for improvement
            if val_loss < self.history['best_val_loss']:
                self.history['best_val_loss'] = val_loss
                self.history['best_epoch'] = epoch
                self._save_checkpoint(epoch, is_best=True)
                self.patience_counter = 0
                self.logger.info(f"New best model at epoch {epoch+1}")
            else:
                self.patience_counter += 1
            
            # Logging
            if epoch % 10 == 0 or epoch == epochs - 1:
                self.logger.info(
                    f"Epoch {epoch+1:3d}/{epochs}: "
                    f"Train Loss = {train_loss:.4f}, "
                    f"Val Loss = {val_loss:.4f}, "
                    f"Best = {self.history['best_val_loss']:.4f}"
                )
            
            # Early stopping
            if self.patience_counter >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self._load_best_checkpoint()
        
        self.logger.info(
            f"Training completed. Best validation loss: {self.history['best_val_loss']:.4f} "
            f"at epoch {self.history['best_epoch']+1}"
        )
        
        return self.history
    
    def _create_dataloader(self, data: torch.Tensor, shuffle: bool = True) -> DataLoader:
        """Create PyTorch DataLoader."""
        dataset = TensorDataset(data)
        return DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=shuffle,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
    
    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data,) in enumerate(dataloader):
            data = data.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            loss, pred, mask = self.model(
                data, 
                mask_ratio=self.config.get('mask_ratio', 0.75)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (data,) in enumerate(dataloader):
                data = data.to(self.device)
                
                loss, pred, mask = self.model(
                    data, 
                    mask_ratio=self.config.get('mask_ratio', 0.75)
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'outputs/models'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
    
    def _load_best_checkpoint(self):
        """Load the best checkpoint."""
        best_path = Path(self.config.get('checkpoint_dir', 'outputs/models')) / 'best_model.pt'
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("Loaded best model checkpoint")
    
    def save_final_model(self, save_path: str):
        """Save the final trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': self.history
        }, save_path)
        self.logger.info(f"Final model saved to {save_path}")
