"""
Calculate Dataset Statistics.

Utility script to calculate mean and standard deviation for image datasets.
These values are used for normalization in data preprocessing.

Usage:
    python scripts/calculate_dataset_stats.py \
        --config configs/flowers17_minivggnet.yaml

Output:
    Prints mean and std values that can be copied to config files.
"""

import argparse

import torch

from data import get_dataset_for_stats
from utils import Config


def main():
    """Calculate and print dataset statistics from configuration file."""
    parser = argparse.ArgumentParser(
        description='Calculate mean and std for image dataset'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    args = parser.parse_args()

    print(f'Loading configuration from: {args.config}')
    cfg = Config(args.config)

    # Get full dataset loader for statistics calculation
    print('Loading dataset for statistics calculation...')
    stats_loader = get_dataset_for_stats(cfg)

    print(f'Calculating statistics for {len(stats_loader.dataset)} images...')
    print(f'Number of classes: {len(stats_loader.dataset.classes)}')
    print(f'Resize: {cfg.transforms.common.resize}, Crop: {cfg.transforms.common.crop}')

    # Initialize variables for online mean/std calculation
    channels_sum = torch.zeros(3)
    channels_squared_sum = torch.zeros(3)
    num_batches = 0

    # Calculate mean and std
    for images, _ in stats_loader:
        # images shape: [batch_size, 3, H, W]
        channels_sum += torch.mean(images, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(images ** 2, dim=[0, 2, 3])
        num_batches += 1

    # Calculate final mean and std
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    mean = mean.tolist()
    std = std.tolist()

    print('\n' + '='*60)
    print('Dataset Statistics')
    print('='*60)
    print(f'Mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]')
    print(f'Std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]')
    print('='*60)
    print('\nAdd to your config file:')
    print('transforms:')
    print('  common:')
    print(f'    normalize:')
    print(f'      mean: [{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}]')
    print(f'      std: [{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]')
    print('='*60)


if __name__ == '__main__':
    main()
