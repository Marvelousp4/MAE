"""
Reproducibility utilities for scientific experiments.
"""

import torch
import numpy as np
import random
import os


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_deterministic():
    """
    Set deterministic behavior for PyTorch operations.
    Note: May impact performance but ensures reproducibility.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device_info():
    """Get device information for reproducibility reporting."""
    if torch.cuda.is_available():
        return {
            'device': 'cuda',
            'gpu_name': torch.cuda.get_device_name(),
            'gpu_count': torch.cuda.device_count(),
            'cuda_version': torch.version.cuda,
            'pytorch_version': torch.__version__
        }
    else:
        return {
            'device': 'cpu',
            'pytorch_version': torch.__version__
        }
