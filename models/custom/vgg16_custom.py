"""
VGG16 Custom Architecture.

PyTorch implementation of VGG16 with batch normalization for training from
scratch. Features Conv -> BN -> ReLU ordering for improved training stability.

Architecture:
    Block 1: [Conv3x3(64) -> BN -> ReLU] * 2 -> MaxPool
    Block 2: [Conv3x3(128) -> BN -> ReLU] * 2 -> MaxPool
    Block 3: [Conv3x3(256) -> BN -> ReLU] * 3 -> MaxPool
    Block 4: [Conv3x3(512) -> BN -> ReLU] * 3 -> MaxPool
    Block 5: [Conv3x3(512) -> BN -> ReLU] * 3 -> MaxPool
    AdaptiveAvgPool2d(7x7)
    FC: [Linear(25088->4096) -> ReLU -> Dropout(0.5) ->
         Linear(4096->4096) -> ReLU -> Dropout(0.5) ->
         Linear(4096->num_classes)]

Key Features:
    - Batch normalization after each convolution for training stability
    - Adaptive pooling for consistent feature map size
    - Dropout regularization (0.5) in fully connected layers
    - Designed for 224x224 RGB input images
    - Trains from scratch (no pretrained weights)

Classes:
    VGG16Custom: PyTorch nn.Module implementation

Example:
    >>> from models.custom.vgg16_custom import VGG16Custom
    >>> model = VGG16Custom(num_classes=1000)
    >>> output = model(torch.randn(1, 3, 224, 224))
"""

import torch
from torch import nn


class VGG16Custom(nn.Module):
    """VGG16 architecture with batch normalization for training from scratch."""

    def __init__(self, num_classes=1000):
        """
        Initialize VGG16Custom for 224x224 RGB images.

        Args:
            num_classes: Number of output classes (default: 1000)
        """
        super(VGG16Custom, self).__init__()
        self.feature = nn.Sequential(
            # Block 1: (CONV => BN => RELU) * 2 => POOL
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: (CONV => BN => RELU) * 2 => POOL
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: (CONV => BN => RELU) * 3 => POOL
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: (CONV => BN => RELU) * 3 => POOL
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5: (CONV => BN => RELU) * 3 => POOL
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Adaptive pooling to ensure 7x7 spatial size
        # Output: 512 channels * 7 * 7 = 25088
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize model weights using best practices for ReLU networks.

        Applies:
        - Kaiming normal initialization for Conv2d layers (optimal for ReLU)
        - Constant initialization for BatchNorm2d (weight=1, bias=0)
        - Normal initialization for Linear layers (mean=0, std=0.01)
        - Constant initialization for all biases (0)
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                       nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.feature(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
