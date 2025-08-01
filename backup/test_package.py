#!/usr/bin/env python3
"""
Simple test script to verify our fMRI-MAE package works.
"""

import sys
import os
import traceback

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """Test all imports."""
    print("🧪 Testing fMRI-MAE Package")
    print("=" * 50)
    
    # Test basic dependencies
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    # Test our package imports
    try:
        from src.models.fmri_mae import MaskedAutoencoderFMRI
        print("✓ MaskedAutoencoderFMRI imported")
    except Exception as e:
        print(f"✗ MaskedAutoencoderFMRI import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from src.models.fmri_masking import FMRISpatiotemporalMasking
        print("✓ FMRISpatiotemporalMasking imported")
    except Exception as e:
        print(f"✗ FMRISpatiotemporalMasking import failed: {e}")
        return False
    
    try:
        from src.utils.config import load_config
        print("✓ load_config imported")
    except Exception as e:
        print(f"✗ load_config import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test model creation."""
    print("\n🤖 Testing Model Creation")
    print("=" * 50)
    
    try:
        import torch
        from src.models.fmri_mae import MaskedAutoencoderFMRI
        
        # Create a small test model
        model = MaskedAutoencoderFMRI(
            num_regions=10,  # Small for testing
            seq_len=20,      # Small for testing
            patch_size_T=10,  # Temporal patch size
            embed_dim=128,
            depth=2,
            num_heads=4
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created with {total_params:,} parameters")
        
        # Test forward pass
        batch_size = 2
        test_input = torch.randn(batch_size, 10, 20)
        
        with torch.no_grad():
            loss, pred, mask = model(test_input)
            print(f"✓ Forward pass successful:")
            print(f"  - Loss: {loss.item():.4f}")
            print(f"  - Prediction shape: {pred.shape}")
            print(f"  - Mask shape: {mask.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation/testing failed: {e}")
        traceback.print_exc()
        return False

def test_config():
    """Test configuration loading."""
    print("\n⚙️  Testing Configuration")
    print("=" * 50)
    
    try:
        from src.utils.config import load_config
        
        # Test with our default config
        config = load_config('configs/default.yaml')
        print("✓ Configuration loaded successfully")
        print(f"  - Model embed_dim: {config['model']['embed_dim']}")
        print(f"  - Training batch_size: {config['training']['batch_size']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 fMRI-MAE Package Test Suite")
    print("=" * 50)
    
    success = True
    
    # Run tests
    success &= test_imports()
    success &= test_model_creation()
    success &= test_config()
    
    # Final result
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Package is working correctly.")
    else:
        print("❌ Some tests failed. Check the output above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
