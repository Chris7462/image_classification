"""
Data Loading Module.

This module provides functions for loading and preparing image classification
datasets with stratified train/validation/test splits. Supports custom datasets
and standard PyTorch datasets.

Key Features:
    - Stratified splitting to maintain class distribution across splits
    - Configurable split ratios from dataset config sections
    - Separate transforms for training (with augmentation) and validation/test
    - DataLoader creation with configurable batch size and workers

Functions:
    _split_dataset: Split dataset indices into train/val/test with
        stratification
    _create_imagefolder_loaders: Create ImageFolder dataset loaders
    _create_cifar10_loaders: Create CIFAR10 dataset loaders
    _create_tiny_imagenet200_loaders: Create Tiny ImageNet-200 dataset loaders
    _create_imagenet_loaders: Create ImageNet dataset loaders
    get_data_loaders: Create train/val/test DataLoaders from config

Example:
    >>> from utils import Config
    >>> from data import get_data_loaders
    >>> cfg = Config('configs/flowers17_vgg.yaml')
    >>> train_loader, val_loader, test_loader = get_data_loaders(cfg)
"""

import os
from pathlib import Path

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets

from PIL import Image

from .transforms import _get_transforms


class Subset(torch.utils.data.Dataset):
    """A subset of a dataset at specified indices, exposing parent attributes."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        # Expose common attributes from parent dataset
        if hasattr(dataset, 'classes'):
            self.classes = dataset.classes
        if hasattr(dataset, 'class_to_idx'):
            self.class_to_idx = dataset.class_to_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


class TinyImageNetDataset(Dataset):
    """
    Tiny ImageNet-200 Dataset.

    Handles the unique structure of Tiny ImageNet where:
    - Training images are in train/n########/images/*.JPEG
    - Validation images are in val/images/*.JPEG with annotations in val_annotations.txt

    Args:
        root: Path to tiny-imagenet-200 directory
        split: 'train', 'val', or 'test'
        transform: Transform to apply to images
        train_test_split_ratio: For 'test' split, ratio of training data to use (default: 0.1)
    """

    def __init__(self, root, split='train', transform=None, train_test_split_ratio=0.1,
                 random_state=42):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.train_test_split_ratio = train_test_split_ratio
        self.random_state = random_state

        # Get all class directories from train folder
        train_dir = self.root / 'train'
        self.classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        self.targets = []

        if split in ['train', 'test']:
            self._load_train_test_data()
        elif split == 'val':
            self._load_val_data()
        else:
            raise ValueError(f"Split must be 'train', 'val', or 'test', got {split}")

    def _load_train_test_data(self):
        """Load training or test data by splitting the training directory."""
        train_dir = self.root / 'train'

        # Collect all samples per class
        all_class_samples = {}
        for class_name in self.classes:
            class_dir = train_dir / class_name / 'images'
            class_samples = []

            if class_dir.exists():
                for img_path in sorted(class_dir.glob('*.JPEG')):
                    class_samples.append((str(img_path), self.class_to_idx[class_name]))

            all_class_samples[class_name] = class_samples

        # Split each class and select appropriate subset based on self.split
        for class_name, class_samples in all_class_samples.items():
            if len(class_samples) > 0:
                # Split this class's samples
                indices = list(range(len(class_samples)))
                train_indices, test_indices = train_test_split(
                    indices,
                    test_size=self.train_test_split_ratio,
                    random_state=self.random_state
                )

                # Select appropriate indices based on split type
                selected_indices = train_indices if self.split == 'train' else test_indices

                # Add selected samples
                for idx in selected_indices:
                    img_path, label = class_samples[idx]
                    self.samples.append(img_path)
                    self.targets.append(label)

    def _load_val_data(self):
        """Load validation data using val_annotations.txt."""
        val_dir = self.root / 'val' / 'images'
        annotations_file = self.root / 'val' / 'val_annotations.txt'

        if not annotations_file.exists():
            raise FileNotFoundError(f"Validation annotations not found at {annotations_file}")

        # Parse annotations file
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_name = parts[0]
                    class_name = parts[1]

                    img_path = val_dir / img_name
                    if img_path.exists() and class_name in self.class_to_idx:
                        self.samples.append(str(img_path))
                        self.targets.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        target = self.targets[idx]

        # Load image
        img = Image.open(img_path).convert('RGB')

        # Apply transform
        if self.transform is not None:
            img = self.transform(img)

        return img, target


def _split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                   random_state=42):
    """
    Split dataset into train/val/test sets with stratification.

    Args:
        dataset: PyTorch dataset with .targets attribute
        train_ratio: Proportion for training set (default: 0.7)
        val_ratio: Proportion for validation set (default: 0.15)
        test_ratio: Proportion for test set (default: 0.15)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        tuple: (train_idx, val_idx, test_idx) - Indices for each split

    Raises:
        AssertionError: If split ratios don't sum to 1.0
    """
    # Validate split ratios
    ratio_sum = train_ratio + val_ratio + test_ratio
    assert abs(ratio_sum - 1.0) < 1e-6, (
        f'Split ratios must sum to 1.0, got {ratio_sum:.6f} '
        f'(train={train_ratio}, val={val_ratio}, test={test_ratio})'
    )

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
    if val_ratio == 0:
        # No separate validation set - use test indices for both
        train_idx = train_val_idx
        val_idx = test_idx  # Point to same data as test
    else:
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_ratio_adjusted,
            random_state=random_state,
            stratify=[dataset.targets[i] for i in train_val_idx]
        )

    return train_idx, val_idx, test_idx


def _create_imagefolder_loaders(cfg, train_tf, val_tf, test_tf):
    """
    Create DataLoaders for ImageFolder-based datasets from config.

    Works for any dataset organized in ImageFolder format (e.g., Flowers17,
    Animals).

    Args:
        cfg: Configuration object with dataset section
        train_tf: Training transforms
        val_tf: Validation transforms
        test_tf: Test transforms

    Returns:
        tuple: (train_set, val_set, test_set) - Dataset subsets

    Raises:
        FileNotFoundError: If dataset path does not exist
    """
    data_root = cfg.dataset.data_root

    if not os.path.exists(data_root):
        raise FileNotFoundError(f'Dataset not found at: {data_root}')

    # Load the full dataset for splitting
    full_dataset = datasets.ImageFolder(root=data_root)

    # Split into train/val/test with configured ratios
    train_idx, val_idx, test_idx = _split_dataset(
        full_dataset,
        train_ratio=cfg.dataset.split_ratios.train,
        val_ratio=cfg.dataset.split_ratios.val,
        test_ratio=cfg.dataset.split_ratios.test,
        random_state=cfg.random_seed
    )

    # Create subset datasets with appropriate transforms
    train_set = Subset(datasets.ImageFolder(
        root=data_root, transform=train_tf), train_idx)
    val_set = Subset(datasets.ImageFolder(
        root=data_root, transform=val_tf), val_idx)
    test_set = Subset(datasets.ImageFolder(
        root=data_root, transform=test_tf), test_idx)

    return train_set, val_set, test_set


def _create_cifar10_loaders(cfg, train_tf, val_tf):
    """
    Create DataLoaders for CIFAR10 dataset from config.

    Args:
        cfg: Configuration object with optional dataset section
        train_tf: Training transforms
        val_tf: Validation transforms
        test_tf: Test transforms

    Returns:
        tuple: (train_set, val_set, test_set) - Datasets

    Note:
        CIFAR10 uses the standard train/test split from torchvision.
        Validation and test sets are identical (official test set).
        If dataset.download_path is not specified, defaults to './data'.
    """
    # Get download path from config, or use default
    if hasattr(cfg.dataset, 'download_path'):
        download_path = cfg.dataset.download_path
    else:
        download_path = './datasets'

    train_set = datasets.CIFAR10(root=download_path, train=True,
                                 transform=train_tf, download=True)
    val_set = datasets.CIFAR10(root=download_path, train=False,
                               transform=val_tf)
    test_set = val_set  # For CIFAR10, val and test are the same

    return train_set, val_set, test_set


def _create_tiny_imagenet200_loaders(cfg, train_tf, val_tf, test_tf):
    """
    Create DataLoaders for Tiny ImageNet-200 dataset from config.

    Args:
        cfg: Configuration object with dataset section
        train_tf: Training transforms
        val_tf: Validation transforms
        test_tf: Test transforms

    Returns:
        tuple: (train_set, val_set, test_set) - Dataset objects

    Raises:
        FileNotFoundError: If dataset path does not exist
    """
    data_root = cfg.dataset.data_root

    if not os.path.exists(data_root):
        raise FileNotFoundError(f'Dataset not found at: {data_root}')

    # Get train_test_split_ratio from config, default to 0.1 (50 images per class)
    # 50 images / 500 images = 0.1
    if hasattr(cfg.dataset, 'train_test_split_ratio'):
        train_test_split_ratio = cfg.dataset.train_test_split_ratio
    else:
        train_test_split_ratio = 0.1

    # Create datasets for each split
    train_set = TinyImageNetDataset(
        root=data_root,
        split='train',
        transform=train_tf,
        train_test_split_ratio=train_test_split_ratio,
        random_state=cfg.random_seed
    )

    val_set = TinyImageNetDataset(
        root=data_root,
        split='val',
        transform=val_tf
    )

    test_set = TinyImageNetDataset(
        root=data_root,
        split='test',
        transform=test_tf,
        train_test_split_ratio=train_test_split_ratio,
        random_state=cfg.random_seed
    )

    return train_set, val_set, test_set


def _create_imagenet_loaders(cfg, train_tf, val_tf, test_tf):
    """
    Create DataLoaders for ImageNet dataset from config.

    Uses the standard ImageNet train/val folder structure. Splits the training
    folder into 90% train and 10% test, while using the validation folder
    as the validation set.

    Args:
        cfg: Configuration object with dataset section
        train_tf: Training transforms
        val_tf: Validation transforms
        test_tf: Test transforms

    Returns:
        tuple: (train_set, val_set, test_set) - Dataset objects

    Raises:
        FileNotFoundError: If dataset path does not exist
    """
    data_root = cfg.dataset.data_root

    if not os.path.exists(data_root):
        raise FileNotFoundError(f'Dataset not found at: {data_root}')

    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f'Training directory not found at: {train_dir}')
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f'Validation directory not found at: {val_dir}')

    # Load full training dataset for splitting
    full_train_dataset = datasets.ImageFolder(root=train_dir)

    # Split training data into 90% train, 10% test with stratification
    train_idx, test_idx = train_test_split(
        list(range(len(full_train_dataset))),
        test_size=0.1,
        random_state=cfg.random_seed,
        stratify=full_train_dataset.targets
    )

    # Create train and test datasets with appropriate transforms
    train_set = Subset(datasets.ImageFolder(
        root=train_dir, transform=train_tf), train_idx)
    test_set = Subset(datasets.ImageFolder(
        root=train_dir, transform=test_tf), test_idx)

    # Load validation dataset separately (no splitting)
    val_set = datasets.ImageFolder(root=val_dir, transform=val_tf)

    return train_set, val_set, test_set


def get_data_loaders(cfg):
    """
    Create train, validation, and test DataLoaders from configuration.

    Args:
        cfg: Configuration object containing:
            - random_seed: Random seed for reproducibility
            - dataset.name: Dataset name ('flowers17', 'animals', 'cifar10',
              'tiny_imagenet200', 'imagenet', etc.)
            - dataset.data_root: Path to dataset
            - dataset.split_ratios: Train/val/test split ratios (not used for
              tiny_imagenet200 or imagenet)
            - data_loader.batch_size: Batch size for DataLoaders
            - data_loader.num_workers: Number of worker processes
            - transforms: Transform configuration for train/val/test

    Returns:
        tuple: (train_loader, val_loader, test_loader) - DataLoader objects

    Raises:
        ValueError: If dataset name is not supported
        FileNotFoundError: If dataset path does not exist

    Example:
        >>> cfg = Config('configs/flowers17_vgg.yaml')
        >>> train_loader, val_loader, test_loader = get_data_loaders(cfg)
        >>> print(f"Training batches: {len(train_loader)}")
    """
    train_tf, val_tf, test_tf = _get_transforms(cfg.transforms)
    dataset_name = cfg.dataset.name.lower()

    # Create datasets based on name
    if dataset_name in ['animals', 'caltech-101', 'dogs_vs_cats', 'flowers17']:
        train_set, val_set, test_set = \
            _create_imagefolder_loaders(cfg, train_tf, val_tf, test_tf)
    elif dataset_name == 'cifar10':
        train_set, val_set, test_set = \
            _create_cifar10_loaders(cfg, train_tf, val_tf)
    elif dataset_name == 'tiny_imagenet200':
        train_set, val_set, test_set = \
            _create_tiny_imagenet200_loaders(cfg, train_tf, val_tf, test_tf)
    elif dataset_name == 'imagenet':
        train_set, val_set, test_set = \
            _create_imagenet_loaders(cfg, train_tf, val_tf, test_tf)
    else:
        raise ValueError(
            f"Unsupported dataset: '{cfg.dataset.name}'."
        )

    # Create DataLoaders with common parameters
    loader_kwargs = {
        'batch_size': cfg.data_loader.batch_size,
        'num_workers': cfg.data_loader.num_workers,
        'pin_memory': True
    }

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


def get_dataset_for_stats(cfg):
    """
    Create a DataLoader for calculating dataset statistics.

    This function loads the full training dataset (before any splits) with
    minimal transforms: Resize -> CenterCrop -> ToTensor (no normalization).
    This is used for calculating mean and std values for normalization.

    Args:
        cfg: Configuration object containing:
            - dataset.name: Dataset name
            - dataset.data_root or dataset.download_path: Path to dataset
            - transforms.common.resize: Resize dimension
            - transforms.common.crop: Crop dimension
            - data_loader.batch_size: Batch size
            - data_loader.num_workers: Number of workers

    Returns:
        DataLoader: DataLoader with full dataset for statistics calculation

    Raises:
        ValueError: If dataset name is not supported
        FileNotFoundError: If dataset path does not exist

    Example:
        >>> cfg = Config('configs/flowers17_vgg.yaml')
        >>> stats_loader = get_dataset_for_stats(cfg)
        >>> # Calculate mean and std from stats_loader
    """
    from torchvision import transforms

    # Create minimal transform: Resize -> CenterCrop -> ToTensor (no normalization)
    stats_transform = transforms.Compose([
        transforms.Resize(cfg.transforms.common.resize),
        transforms.CenterCrop(cfg.transforms.common.crop),
        transforms.ToTensor()
    ])

    dataset_name = cfg.dataset.name.lower()

    # Load full dataset based on dataset type
    if dataset_name in ['animals', 'caltech-101', 'dogs_vs_cats', 'flowers17']:
        data_root = cfg.dataset.data_root
        if not os.path.exists(data_root):
            raise FileNotFoundError(f'Dataset not found at: {data_root}')
        full_dataset = datasets.ImageFolder(root=data_root, transform=stats_transform)

    elif dataset_name == 'cifar10':
        download_path = getattr(cfg.dataset, 'download_path', './datasets')
        # Use training set for CIFAR10 (50k images)
        full_dataset = datasets.CIFAR10(
            root=download_path,
            train=True,
            transform=stats_transform,
            download=True
        )

    elif dataset_name == 'tiny_imagenet200':
        data_root = cfg.dataset.data_root
        if not os.path.exists(data_root):
            raise FileNotFoundError(f'Dataset not found at: {data_root}')
        # Use full training directory (all 500 images per class)
        train_dir = Path(data_root) / 'train'
        full_dataset = datasets.ImageFolder(root=str(train_dir), transform=stats_transform)

    elif dataset_name == 'imagenet':
        data_root = cfg.dataset.data_root
        if not os.path.exists(data_root):
            raise FileNotFoundError(f'Dataset not found at: {data_root}')
        # Use full training directory
        train_dir = os.path.join(data_root, 'train')
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f'Training directory not found at: {train_dir}')
        full_dataset = datasets.ImageFolder(root=train_dir, transform=stats_transform)

    else:
        raise ValueError(f"Unsupported dataset: '{cfg.dataset.name}'")

    # Create DataLoader
    loader_kwargs = {
        'batch_size': cfg.data_loader.batch_size,
        'num_workers': cfg.data_loader.num_workers,
        'pin_memory': True,
        'shuffle': False
    }

    stats_loader = DataLoader(full_dataset, **loader_kwargs)

    return stats_loader
