"""
MiniGoogLeNet Architecture.

A lightweight implementation of GoogLeNet/Inception architecture for image
classification. Features inception modules with parallel 1x1 and 3x3
convolutions, downsample modules for spatial reduction, and global average
pooling for classification.

Architecture:
    Conv(96) -> 2xInception -> Downsample -> 4xInception -> Downsample ->
    2xInception -> AdaptiveAvgPool -> Dropout(0.5) -> FC(num_classes)

Key Features:
    - Inception modules with parallel multi-scale convolutions
    - Downsample modules combining convolution and max pooling
    - Adaptive pooling for flexible input image sizes
    - Batch normalization for training stability
    - Dropout regularization to prevent overfitting

Classes:
    ConvModule: Basic convolution block with BN and ReLU
    InceptionModule: Parallel 1x1 and 3x3 convolution branches
    DownsampleModule: Spatial reduction with conv and pooling branches
    MiniGoogLeNet: Complete architecture

Example:
    >>> from models.custom.minigooglenet import MiniGoogLeNet
    >>> model = MiniGoogLeNet(num_classes=10)
    >>> output = model(torch.randn(1, 3, 64, 64))  # Flexible input size
"""

import torch
from torch import nn


class ConvModule(nn.Module):
    """
    Basic convolution block: Conv2d -> BatchNorm2d -> ReLU.

    This module provides a reusable building block for the network with
    proper normalization and activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Initialize convolution module.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel
            stride: Stride for convolution
            padding: Padding for convolution
                - For "same" padding: kernel_size//2 (if kernel_size is odd)
                - For "valid" padding: 0
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass through Conv -> BN -> ReLU.

        Args:
            x: Input tensor

        Returns:
            Output tensor after convolution, normalization, and activation
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionModule(nn.Module):
    """
    Inception module with parallel 1x1 and 3x3 convolution branches.

    Applies two parallel convolution paths and concatenates their outputs
    along the channel dimension, enabling multi-scale feature extraction.
    """

    def __init__(self, in_channels, numK1x1, numK3x3):
        """
        Initialize inception module with two branches.

        Args:
            in_channels: Number of input channels
            numK1x1: Number of filters in 1x1 convolution branch
            numK3x3: Number of filters in 3x3 convolution branch
        """
        super().__init__()
        # 1x1 branch: no padding needed (kernel 1x1 preserves dimensions)
        self.branch1 = ConvModule(in_channels, numK1x1, kernel_size=1,
                                  stride=1, padding=0)
        # 3x3 branch: padding=1 for "same" padding
        self.branch3 = ConvModule(in_channels, numK3x3, kernel_size=3,
                                  stride=1, padding=1)

    def forward(self, x):
        """
        Forward pass through parallel branches with concatenation.

        Args:
            x: Input tensor

        Returns:
            Concatenated output from both branches along channel dimension
        """
        out1 = self.branch1(x)
        out3 = self.branch3(x)
        # Concatenate along channel dimension (dim=1)
        return torch.cat([out1, out3], dim=1)


class DownsampleModule(nn.Module):
    """
    Downsample module with parallel convolution and pooling branches.

    Reduces spatial dimensions using both learned convolution and max pooling,
    concatenating the results for richer feature representation.
    """

    def __init__(self, in_channels, K):
        """
        Initialize downsample module.

        Args:
            in_channels: Number of input channels
            K: Number of filters in convolution branch
        """
        super().__init__()
        # Convolution branch: 3x3 kernel, stride 2, "valid" padding
        self.conv = ConvModule(in_channels, K, kernel_size=3, stride=2,
                               padding=0)
        # Pooling branch: 3x3 kernel, stride 2
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        """
        Forward pass through parallel conv and pooling with concatenation.

        Args:
            x: Input tensor

        Returns:
            Concatenated output from conv and pooling branches
        """
        conv_out = self.conv(x)
        pool_out = self.pool(x)
        return torch.cat([conv_out, pool_out], dim=1)


class MiniGoogLeNet(nn.Module):
    """
    MiniGoogLeNet architecture with inception modules for image classification.

    A compact implementation of GoogLeNet/Inception that uses parallel
    convolutions at multiple scales, downsample modules for spatial reduction,
    and adaptive pooling for flexible input sizes.
    """

    def __init__(self, num_classes):
        """
        Initialize MiniGoogLeNet for RGB images with flexible input size.

        Args:
            num_classes: Number of output classes
        """
        super().__init__()

        # 1. First CONV module: 96 filters, 3x3, stride 1, same padding
        self.conv1 = ConvModule(3, 96, kernel_size=3, stride=1, padding=1)

        # 2. Two Inception modules:
        # 96 -> (32 + 32) = 64 channels
        self.incept1 = InceptionModule(96, 32, 32)
        # 64 -> (32 + 48) = 80 channels
        self.incept2 = InceptionModule(64, 32, 48)

        # 3. First Downsample: 80 -> (80 + 80) = 160 channels
        self.down1 = DownsampleModule(80, 80)

        # 4. Four Inception modules:
        # 160 -> (112 + 48) = 160 channels
        self.incept3 = InceptionModule(160, 112, 48)
        # 160 -> (96 + 64) = 160 channels
        self.incept4 = InceptionModule(160, 96, 64)
        # 160 -> (80 + 80) = 160 channels
        self.incept5 = InceptionModule(160, 80, 80)
        # 160 -> (48 + 96) = 144 channels
        self.incept6 = InceptionModule(160, 48, 96)

        # 5. Second Downsample: 144 -> (96 + 144) = 240 channels
        self.down2 = DownsampleModule(144, 96)

        # 6. Two Inception modules:
        # 240 -> (176 + 160) = 336 channels
        self.incept7 = InceptionModule(240, 176, 160)
        # 336 -> (176 + 160) = 336 channels
        self.incept8 = InceptionModule(336, 176, 160)

        # 7. Global Average Pooling: flexible input size -> 1x1 output
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 8. Dropout layer with probability 0.5
        self.dropout = nn.Dropout(0.5)

        # 9. Fully connected layer: 336 channels -> num_classes
        # After adaptive pooling: (batch, 336, 1, 1) -> flatten to 336
        self.fc = nn.Linear(336, num_classes)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.incept1(x)
        x = self.incept2(x)
        x = self.down1(x)
        x = self.incept3(x)
        x = self.incept4(x)
        x = self.incept5(x)
        x = self.incept6(x)
        x = self.down2(x)
        x = self.incept7(x)
        x = self.incept8(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
