"""
Data Module

Public API for data loading and preprocessing functionality. This module provides
convenient access to data loaders and image transformations for training and
evaluation.

Exported Functions:
    get_data_loaders: Create train/val/test DataLoaders from configuration
    get_transforms: Build training and validation transform pipelines

Usage:
    >>> from data import get_data_loaders, get_transforms
    >>> train_loader, val_loader, test_loader = get_data_loaders(cfg)
"""

# Import main classes and functions
from .data_loaders import get_data_loaders
from .transforms import get_transforms

__all__ = [
    'get_data_loaders',
    'get_transforms'
]
