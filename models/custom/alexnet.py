"""
AlexNet Architecture.

PyTorch implementation of AlexNet for image classification, designed for
227x227 input images. Features batch normalization after each layer for
improved training stability.

Architecture:
    Block 1: [Conv11x11(96) -> BN -> ReLU -> MaxPool -> Dropout(0.25)]
    Block 2: [Conv5x5(256) -> BN -> ReLU -> MaxPool -> Dropout(0.25)]
    Block 3: [Conv3x3(384) -> BN -> ReLU -> Conv3x3(384) -> BN -> ReLU ->
        Conv3x3(256) -> BN -> ReLU -> MaxPool -> Dropout(0.25)]
    FC: [Flatten -> Linear(4096) -> BN -> ReLU -> Dropout(0.5) ->
        Linear(4096) -> BN -> ReLU -> Dropout(0.5) -> Linear(num_classes)]

Key Features:
    - Batch normalization for training stability
    - Dropout regularization (0.25 for conv, 0.5 for FC)
    - Dynamic feature size calculation
    - Designed for 227x227 RGB input images

Classes:
    AlexNet: PyTorch nn.Module implementation

Example:
    >>> from models.custom.alexnet import AlexNet
    >>> model = AlexNet(num_classes=17)
    >>> output = model(torch.randn(1, 3, 227, 227))
"""

import torch
from torch import nn


class AlexNet(nn.Module):
    """AlexNet architecture with batch normalization."""

    def __init__(self, num_classes):
        """
        Initialize AlexNet for 227x227 RGB images.

        Args:
            num_classes: Number of output classes
        """
        super().__init__()

        # Feature extractor with manually computed padding values
        self.features = nn.Sequential(
            # Block 1: Conv => BatchNorm => ReLU => MaxPool => Dropout
            # For input 227x227, using kernel_size=11, stride=4, padding=4
            # Output: (227 + 2*4 - 11) / 4 + 1 = 57
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11,
                      stride=4, padding=4),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.25),

            # Block 2: Conv => BatchNorm => ReLU => MaxPool => Dropout
            # For kernel=5, stride=1, "same" padding=2
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5,
                      stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.25),

            # Block 3: Three conv layers with kernel=3, stride=1, padding=1
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.25)
        )

        # Compute the flattened feature size dynamically using a dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 227, 227)
            dummy_out = self.features(dummy_input)
            self.flatten_dim = dummy_out.view(1, -1).size(1)

        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Block 4: FC => BN => ReLU => Dropout
            nn.Linear(self.flatten_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # Block 5: FC => BN ReLU => => Dropout
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # Output logits for softmax classifier
            nn.Linear(4096, num_classes),
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
