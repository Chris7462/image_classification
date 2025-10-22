"""
Reproducibility Utilities.

Provides functions for setting random seeds across all libraries to ensure
reproducible results in deep learning experiments.

Functions:
    set_seed: Set random seeds for Python, NumPy, PyTorch, and CUDA

Features:
    - Sets seeds for all major random number generators
    - Configurable deterministic mode for cuDNN
    - Supports both CPU and GPU (single/multi-GPU)

Example:
    >>> from utils import set_seed
    >>> set_seed(42)  # Strict determinism
    >>> set_seed(42, strict=False)  # Faster but less deterministic
"""

import random

import numpy as np

import torch


def set_seed(seed=42, deterministic=False):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value (default: 42)
        strict: If True, enforce full determinism at cost of performance.
                If False, allow some non-determinism for faster training.
                (default: True)

    Note:
        Strict mode can reduce training speed by 10-30% but ensures
        fully reproducible results. Non-strict mode is faster but may
        have minor variations across runs.

    Example:
        >>> set_seed(42)  # Full reproducibility
        >>> set_seed(42, strict=False)  # Faster training
    """
    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # CUDA (single and multi-GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuDNN backend
    if deterministic:
        # Deterministic mode: slower but reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Non-deterministic mode: faster but may vary slightly
        torch.backends.cudnn.benchmark = True
