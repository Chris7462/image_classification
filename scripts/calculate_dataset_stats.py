"""
Calculate Dataset Statistics

Utility script to calculate mean and standard deviation for image datasets.
These values are used for normalization in data preprocessing.

Usage:
    python utils/calculate_dataset_stats.py --config configs/flowers17_minivggnet.yaml

Output:
    Prints mean and std values that can be copied to config files.
"""

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import Config


def calculate_mean_std(data_root, resize_size, crop_size, batch_size=64, num_workers=4):
    """
    Calculate mean and standard deviation of RGB channels for a dataset.

    Args:
        data_root: Path to dataset root directory (ImageFolder format)
        resize_size: Size to resize images to
        crop_size: Size to crop images to
        batch_size: Batch size for data loading
        num_workers: Number of worker processes

    Returns:
        tuple: (mean, std) - Lists of RGB channel statistics
    """
    # Use simple transforms - just convert to tensor
    # Match the transforms that will be used in training (without augmentation)
    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Calculating statistics for {len(dataset)} images...")
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"Resize: {resize_size}, Crop: {crop_size}")

    # Initialize variables for online mean/std calculation
    channels_sum = torch.zeros(3)
    channels_squared_sum = torch.zeros(3)
    num_batches = 0

    # Calculate mean and std
    for images, _ in loader:
        # images shape: [batch_size, 3, H, W]
        # Calculate mean and std per channel
        channels_sum += torch.mean(images, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(images ** 2, dim=[0, 2, 3])
        num_batches += 1

    # Calculate final mean and std
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean.tolist(), std.tolist()


def main():
    parser = argparse.ArgumentParser(description='Calculate mean and std for image dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()

    print(f"Loading configuration from: {args.config}")
    cfg = Config(args.config)

    # Load all parameters from config using dot notation
    data_root = cfg.dataset.config.data_root
    resize_size = cfg.dataset.transforms.resize
    crop_size = cfg.dataset.transforms.crop
    batch_size = cfg.dataset.batch_size
    num_workers = cfg.dataset.num_workers

    try:
        mean, std = calculate_mean_std(data_root, resize_size, crop_size, batch_size, num_workers)

        print("\n" + "="*60)
        print("Dataset Statistics")
        print("="*60)
        print(f"Mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
        print(f"Std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
        print("="*60)
        print("\nAdd to your config file:")
        print("transforms:")
        print("  normalize:")
        print(f"    mean: [{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}]")
        print(f"    std: [{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]")
        print("="*60)

    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_root}")
        print("Please check the path and try again.")
    except KeyError as e:
        print(f"Error: Missing key in config file: {e}")
        print("Please check your config file structure.")
    except Exception as e:
        print(f"Error calculating statistics: {e}")


if __name__ == "__main__":
    main()
