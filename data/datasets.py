import os
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.model_selection import train_test_split
from .transforms import get_transforms

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Split dataset into train/val/test sets.

    Args:
        dataset: PyTorch dataset
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_state: Random seed for reproducibility

    Returns:
        train_idx, val_idx, test_idx: Indices for each split
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

def get_data_loaders(cfg):
    train_tf, val_tf = get_transforms(cfg.dataset.transforms)
    name = cfg.dataset.name.lower()
    batch_size = cfg.dataset.batch_size
    num_workers = cfg.dataset.num_workers

    if name == "flowers17":
        # Load full dataset from local folder
        data_root = "/data/kaggle/flowers17"

        if not os.path.exists(data_root):
            raise FileNotFoundError(f"Dataset not found at {data_root}")

        # Load the full dataset
        full_dataset = datasets.ImageFolder(root=data_root)

        # Split into train/val/test
        train_idx, val_idx, test_idx = split_dataset(
            full_dataset,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42
        )

        # Create subset datasets with appropriate transforms
        train_set = datasets.ImageFolder(root=data_root, transform=train_tf)
        val_set = datasets.ImageFolder(root=data_root, transform=val_tf)
        test_set = datasets.ImageFolder(root=data_root, transform=val_tf)

        train_set = Subset(train_set, train_idx)
        val_set = Subset(val_set, val_idx)
        test_set = Subset(test_set, test_idx)

    elif name == "cifar10":
        train_set = datasets.CIFAR10(root="./data", train=True, transform=train_tf, download=True)
        val_set = datasets.CIFAR10(root="./data", train=False, transform=val_tf)
        test_set = val_set
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
