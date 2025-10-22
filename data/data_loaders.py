"""
Data Loading Module

This module provides functions for loading and preparing image classification datasets with
stratified train/validation/test splits. Supports custom datasets and standard PyTorch datasets.

Key Features:
    - Stratified splitting to maintain class distribution across splits
    - Configurable split ratios (default: 70% train, 15% val, 15% test)
    - Separate transforms for training (with augmentation) and validation/test
    - DataLoader creation with configurable batch size and workers

Functions:
    _split_dataset: Split dataset indices into train/val/test with stratification
    get_data_loaders: Create train/val/test DataLoaders from config

Example:
    >>> from utils import Config
    >>> from data import get_data_loaders
    >>> cfg = Config('configs/flowers17_vgg.yaml')
    >>> train_loader, val_loader, test_loader = get_data_loaders(cfg)  # type: DataLoader
"""

import os
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.model_selection import train_test_split
from .transforms import get_transforms


def _split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Split dataset into train/val/test sets.

    Args:
        dataset: PyTorch dataset
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_state: Random seed for reproducibility

    Returns:
        tuple: (train_idx, val_idx, test_idx) - Indices for each split
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    num_samples = len(dataset)
    indices = list(range(num_samples))

    # First split: separate out test set
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=random_state,
        stratify=[dataset.targets[i] for i in indices]
    )

    # Second split: separate train and val from remaining data
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        stratify=[dataset.targets[i] for i in train_val_idx]
    )

    return train_idx, val_idx, test_idx


def _create_flowers17_loaders(cfg, train_tf, val_tf):
    """
    Create DataLoaders for Flowers17 dataset.

    Args:
        cfg: Configuration object
        train_tf: Training transforms
        val_tf: Validation/test transforms

    Returns:
        tuple: (train_set, val_set, test_set) - Dataset subsets
    """
    data_root = "/data/kaggle/flowers17"

    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Dataset not found at {data_root}")

    # Load the full dataset for splitting
    full_dataset = datasets.ImageFolder(root=data_root)

    # Split into train/val/test
    train_idx, val_idx, test_idx = _split_dataset(
        full_dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )

    # Create subset datasets with appropriate transforms
    train_set = Subset(datasets.ImageFolder(root=data_root, transform=train_tf), train_idx)
    val_set = Subset(datasets.ImageFolder(root=data_root, transform=val_tf), val_idx)
    test_set = Subset(datasets.ImageFolder(root=data_root, transform=val_tf), test_idx)

    return train_set, val_set, test_set


def _create_cifar10_loaders(train_tf, val_tf):
    """
    Create DataLoaders for CIFAR10 dataset.

    Args:
        train_tf: Training transforms
        val_tf: Validation/test transforms

    Returns:
        tuple: (train_set, val_set, test_set) - Datasets
    """
    train_set = datasets.CIFAR10(root="./data", train=True, transform=train_tf, download=True)
    val_set = datasets.CIFAR10(root="./data", train=False, transform=val_tf)
    test_set = val_set
    return train_set, val_set, test_set


def get_data_loaders(cfg):
    """
    Create train, validation, and test DataLoaders from configuration.

    Args:
        cfg: Configuration object containing dataset parameters including:
            - dataset.name: Dataset name ('flowers17' or 'cifar10')
            - dataset.batch_size: Batch size for DataLoaders
            - dataset.num_workers: Number of worker processes
            - dataset.transforms: Transform configuration

    Returns:
        tuple: (train_loader, val_loader, test_loader) - DataLoader objects

    Raises:
        ValueError: If dataset name is not supported
        FileNotFoundError: If dataset path does not exist

    Example:
        >>> cfg = Config('configs/flowers17_vgg.yaml')
        >>> train_loader, val_loader, test_loader = get_data_loaders(cfg)
    """
    train_tf, val_tf = get_transforms(cfg.dataset.transforms)
    dataset_name = cfg.dataset.name.lower()

    # Create datasets based on name
    if dataset_name == "flowers17":
        train_set, val_set, test_set = _create_flowers17_loaders(cfg, train_tf, val_tf)
    elif dataset_name == "cifar10":
        train_set, val_set, test_set = _create_cifar10_loaders(train_tf, val_tf)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")

    # Create DataLoaders with common parameters
    loader_kwargs = {
        'batch_size': cfg.dataset.batch_size,
        'num_workers': cfg.dataset.num_workers,
        'pin_memory': True
    }

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
