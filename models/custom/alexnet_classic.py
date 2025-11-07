"""
AlexNet Architecture (Classic Version).

PyTorch implementation of the original AlexNet from "ImageNet Classification 
with Deep Convolutional Neural Networks" (Krizhevsky et al., 2012). This is 
the classic version without batch normalization, matching the original paper.

Architecture:
    Block 1: [Conv11x11(96) -> ReLU -> MaxPool]
    Block 2: [Conv5x5(256) -> ReLU -> MaxPool]
    Block 3: [Conv3x3(384) -> ReLU -> Conv3x3(384) -> ReLU ->
        Conv3x3(256) -> ReLU -> MaxPool]
    FC: [Flatten -> Dropout(0.5) -> Linear(4096) -> ReLU -> 
        Dropout(0.5) -> Linear(4096) -> ReLU -> Dropout(0.5) -> 
        Linear(num_classes)]

Key Features:
    - No batch normalization (original paper design)
    - Dropout only in FC layers (0.5)
    - Dynamic feature size calculation
    - Designed for 227x227 RGB input images

Classes:
    AlexNet: PyTorch nn.Module implementation

Example:
    >>> from models.custom.alexnet import AlexNet
    >>> model = AlexNet(num_classes=1000)
    >>> output = model(torch.randn(1, 3, 227, 227))
"""

import torch
from torch import nn


class AlexNetClassic(nn.Module):
    """Classic AlexNet architecture (original 2012 paper)."""

    def __init__(self, num_classes):
        """
        Initialize classic AlexNet for 227x227 RGB images.

        Args:
            num_classes: Number of output classes
        """
        super().__init__()

        # Feature extractor - convolutional layers
        self.features = nn.Sequential(
            # Block 1: Conv11x11(96) -> ReLU -> MaxPool
            # Input: 227x227x3 -> Output: 27x27x96
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11,
                      stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Block 2: Conv5x5(256) -> ReLU -> MaxPool
            # Input: 27x27x96 -> Output: 13x13x256
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5,
                      stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Block 3: Conv3x3(384) -> ReLU
            # Input: 13x13x256 -> Output: 13x13x384
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv3x3(384) -> ReLU
            # Input: 13x13x384 -> Output: 13x13x384
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv3x3(256) -> ReLU -> MaxPool
            # Input: 13x13x384 -> Output: 6x6x256
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # Compute the flattened feature size dynamically using a dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 227, 227)
            dummy_out = self.features(dummy_input)
            self.flatten_dim = dummy_out.view(1, -1).size(1)

        # Fully connected classifier
        # Original paper: 6x6x256 = 9216 -> 4096 -> 4096 -> num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            
            # FC1: Dropout -> Linear(4096) -> ReLU
            nn.Dropout(p=0.5),
            nn.Linear(self.flatten_dim, 4096),
            nn.ReLU(inplace=True),

            # FC2: Dropout -> Linear(4096) -> ReLU
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            # FC3: Dropout -> Linear(num_classes)
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 227, 227)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
