"""
MiniVGGNet Architecture

Lightweight CNN architecture inspired by VGG, designed for efficient training
on small image datasets. Features two convolutional blocks with batch normalization
and dropout for regularization, followed by fully connected layers.

Architecture:
    Block 1: [Conv3x3(32) -> BN -> ReLU -> Conv3x3(32) -> BN -> ReLU -> MaxPool -> Dropout(0.25)]
    Block 2: [Conv3x3(64) -> BN -> ReLU -> Conv3x3(64) -> BN -> ReLU -> MaxPool -> Dropout(0.25)]
    AdaptiveAvgPool2d(7x7)
    FC: [Flatten -> Linear(3136->512) -> ReLU -> BN -> Dropout(0.5) -> Linear(512->num_classes)]

Key Features:
    - Adaptive pooling for flexible input image sizes
    - Batch normalization for training stability
    - Dropout regularization to prevent overfitting
    - Compact design (~1M parameters for 17 classes)

Classes:
    MiniVGGNet: PyTorch nn.Module implementation

Example:
    >>> from models.custom.minivggnet import MiniVGGNet
    >>> model = MiniVGGNet(num_classes=17)
    >>> output = model(torch.randn(1, 3, 64, 64))  # Flexible input size
"""

import torch.nn as nn


class MiniVGGNet(nn.Module):
    def __init__(self, num_classes):
        """
        MiniVGGNet with adaptive pooling for flexible input sizes.

        Args:
            num_classes: Number of output classes
        """
        super(MiniVGGNet, self).__init__()

        # Block 1: CONV => RELU => CONV => RELU => POOL, with BatchNorm and Dropout.
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25)
        )

        # Block 2: CONV => RELU => CONV => RELU => POOL, with BatchNorm and Dropout.
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
        )

        # Adaptive pooling to fixed spatial size regardless of input dimensions
        # This outputs 64 channels * 7 * 7 spatial dimensions = 3136 features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Calculate flattened dimension: 64 channels * 7 * 7 = 3136
        flattened_dim = 64 * 7 * 7

        # Fully connected layers: Flatten, Dense, ReLU, BatchNorm, Dropout, Dense.
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flattened_dim, out_features=512),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),
            nn.Dropout(p=0.5),
            # logit for softmax classifier
            nn.Linear(in_features=512, out_features=num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.adaptive_pool(x)
        x = self.fc(x)
        return x
