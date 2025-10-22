"""
Data Module.

Public API for data loading and preprocessing functionality. This module
provides convenient access to data loaders for training and evaluation.

Exported Functions:
    get_data_loaders: Create train/val/test DataLoaders from configuration

Usage:
    >>> from data import get_data_loaders
    >>> train_loader, val_loader, test_loader = get_data_loaders(cfg)
"""

# Import main classes and functions
from .data_loaders import get_data_loaders

__all__ = [
    'get_data_loaders'
]
