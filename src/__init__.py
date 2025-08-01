"""
fMRI-MAE: Masked Autoencoder for fMRI Data Analysis

A research-grade implementation of Masked Autoencoders for self-supervised 
learning on functional Magnetic Resonance Imaging (fMRI) data.
"""

__version__ = "0.1.0"
__author__ = "Research Team"
__email__ = "research@example.com"

# Main exports
try:
    from .models.fmri_mae import MaskedAutoencoderFMRI
    from .models.fmri_masking import SpatioTemporalMasking
    from .data.fmri_data_utils import FMRIDataset
    from .training.trainer import MAETrainer
    from .evaluation.downstream_tasks import (
        FNCAnalyzer, 
        DynamicFNCAnalyzer, 
        DiseaseClassifier
    )
    
    __all__ = [
        "MaskedAutoencoderFMRI",
        "SpatioTemporalMasking", 
        "FMRIDataset",
        "MAETrainer",
        "FNCAnalyzer",
        "DynamicFNCAnalyzer",
        "DiseaseClassifier",
    ]
except ImportError:
    # Handle case where modules are not yet available
    __all__ = []

__version__ = "1.0.0"
__author__ = "Research Team"

from .models.fmri_mae import MaskedAutoencoderFMRI, mae_fmri_base, mae_fmri_large
from .models.fmri_masking import FMRISpatiotemporalMasking
from .data.fmri_data_utils import FMRIDataset, create_fmri_dataloader
from .evaluation.downstream_tasks import FNCAnalyzer, DynamicFNCAnalyzer, DiseaseClassifier

__all__ = [
    'MaskedAutoencoderFMRI',
    'mae_fmri_base', 
    'mae_fmri_large',
    'FMRISpatiotemporalMasking',
    'FMRIDataset',
    'create_fmri_dataloader',
    'FNCAnalyzer',
    'DynamicFNCAnalyzer',
    'DiseaseClassifier'
]
