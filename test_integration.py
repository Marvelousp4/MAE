#!/usr/bin/env python3
"""
Complete integration test for fMRI-MAE package.
"""

import sys
import os
import traceback

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_basic_imports():
    """Test basic dependency imports."""
    print("🔧 Testing Basic Dependencies")
    print("=" * 50)
    
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
        
    try:
        import timm
        print(f"✓ TIMM: {timm.__version__}")
    except ImportError as e:
        print(f"✗ TIMM import failed: {e}")
        return False
    
    return True

def test_package_imports():
    """Test our package imports."""
    print("\n🧪 Testing Package Imports")
    print("=" * 50)
    
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
        
    try:
        from src.training.trainer import MAETrainer
        print("✓ MAETrainer imported")
    except Exception as e:
        print(f"✗ MAETrainer import failed: {e}")
        return False
    
    return True

def test_model_functionality():
    """Test model creation and basic functionality."""
    print("\n🤖 Testing Model Functionality")
    print("=" * 50)
    
    try:
        import torch
        from src.models.fmri_mae import MaskedAutoencoderFMRI
        
        # Create test model
        model = MaskedAutoencoderFMRI(
            num_regions=10,
            seq_len=20,
            patch_size_T=10,
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
        print(f"✗ Model testing failed: {e}")
        traceback.print_exc()
        return False

def test_config_system():
    """Test configuration loading."""
    print("\n⚙️  Testing Configuration System")
    print("=" * 50)
    
    try:
        from src.utils.config import load_config
        
        config = load_config('configs/default.yaml')
        print("✓ Configuration loaded successfully")
        print(f"  - Model embed_dim: {config['model']['embed_dim']}")
        print(f"  - Training batch_size: {config['training']['batch_size']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_scripts():
    """Test that scripts can be imported without errors."""
    print("\n📜 Testing Script Imports")
    print("=" * 50)
    
    # We don't actually run the scripts, just check they can be imported
    import subprocess
    import sys
    
    try:
        result = subprocess.run([
            sys.executable, 'scripts/train.py', '--help'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and 'usage:' in result.stdout:
            print("✓ Training script import successful")
        else:
            print(f"✗ Training script failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Training script test failed: {e}")
        return False
    
    try:
        result = subprocess.run([
            sys.executable, 'scripts/evaluate.py', '--help'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and 'usage:' in result.stdout:
            print("✓ Evaluation script import successful")
        else:
            print(f"✗ Evaluation script failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Evaluation script test failed: {e}")
        return False
        
    return True

def main():
    """Run all tests."""
    print("🚀 fMRI-MAE Complete Integration Test")
    print("=" * 60)
    
    success = True
    
    # Run all tests
    success &= test_basic_imports()
    success &= test_package_imports()
    success &= test_model_functionality()
    success &= test_config_system()
    success &= test_scripts()
    
    # Final result
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! Package is fully functional.")
    else:
        print("❌ SOME TESTS FAILED. See output above for details.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
