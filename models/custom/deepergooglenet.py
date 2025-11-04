"""
DeeperGoogLeNet Architecture.

A deeper implementation of GoogLeNet/Inception architecture for image
classification. Features full 4-branch inception modules with parallel 1x1,
3x3, and 5x5 convolutions, plus max pooling branches for multi-scale feature
extraction.

Architecture:
    Conv(64) -> Pool -> Conv(64) -> Conv(192) -> Pool ->
    2xInception -> Pool -> 5xInception -> Pool ->
    AdaptiveAvgPool -> Dropout(0.4) -> FC(num_classes)

Key Features:
    - Full 4-branch Inception modules (1x1, 1x1→3x3, 1x1→5x5, MaxPool→1x1)
    - Dimensionality reduction with 1x1 convolutions before expensive operations
    - Adaptive pooling for flexible input image sizes
    - Batch normalization for training stability
    - Dropout regularization (0.4) to prevent overfitting

Classes:
    ConvModule: Basic convolution block with BN and ReLU
    InceptionModule: Full 4-branch parallel convolution module
    DeeperGoogLeNet: Complete architecture

Example:
    >>> from models.custom.deepergooglenet import DeeperGoogLeNet
    >>> model = DeeperGoogLeNet(num_classes=10)
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
    Full 4-branch Inception module with multi-scale feature extraction.

    Implements four parallel paths:
    1. 1x1 convolution (direct feature extraction)
    2. 1x1 → 3x3 convolution (dimensionality reduction then spatial features)
    3. 1x1 → 5x5 convolution (dimensionality reduction then larger spatial features)
    4. 3x3 MaxPool → 1x1 convolution (pooling features then projection)

    Outputs are concatenated along the channel dimension for rich multi-scale
    representation.
    """

    def __init__(self, in_channels, num1x1, num3x3Reduce, num3x3,
                 num5x5Reduce, num5x5, num1x1Proj):
        """
        Initialize Inception module with four branches.

        Args:
            in_channels: Number of input channels
            num1x1: Number of filters in 1x1 convolution branch
            num3x3Reduce: Number of filters in 1x1 reduction before 3x3 conv
            num3x3: Number of filters in 3x3 convolution
            num5x5Reduce: Number of filters in 1x1 reduction before 5x5 conv
            num5x5: Number of filters in 5x5 convolution
            num1x1Proj: Number of filters in 1x1 projection after max pooling
        """
        super().__init__()

        # Branch 1: Direct 1x1 convolution
        self.branch1 = ConvModule(in_channels, num1x1,
                                  kernel_size=1, stride=1, padding=0)

        # Branch 2: 1x1 reduction followed by 3x3 convolution
        self.branch2 = nn.Sequential(
            ConvModule(in_channels, num3x3Reduce, kernel_size=1, stride=1, padding=0),
            ConvModule(num3x3Reduce, num3x3, kernel_size=3, stride=1, padding=1)
        )

        # Branch 3: 1x1 reduction followed by 5x5 convolution
        self.branch3 = nn.Sequential(
            ConvModule(in_channels, num5x5Reduce, kernel_size=1, stride=1, padding=0),
            ConvModule(num5x5Reduce, num5x5, kernel_size=5, stride=1, padding=2)
        )

        # Branch 4: 3x3 max pooling followed by 1x1 projection
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvModule(in_channels, num1x1Proj, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        """
        Forward pass through all four branches with concatenation.

        Args:
            x: Input tensor

        Returns:
            Concatenated output from all branches along channel dimension
        """
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        # Concatenate along channel dimension (dim=1)
        return torch.cat([out1, out2, out3, out4], dim=1)


class DeeperGoogLeNet(nn.Module):
    """
    DeeperGoogLeNet architecture with full Inception modules.

    A deeper implementation of GoogLeNet that uses full 4-branch Inception
    modules with 1x1, 3x3, and 5x5 convolutions plus max pooling. Uses
    adaptive pooling to support flexible input image sizes.
    """

    def __init__(self, num_classes):
        """
        Initialize DeeperGoogLeNet for RGB images with flexible input size.

        Args:
            num_classes: Number of output classes
        """
        super().__init__()

        # Initial convolutional layers
        self.conv1 = ConvModule(3, 64, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvModule(64, 64, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvModule(64, 192, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 3: Two Inception modules
        # 192 -> (64 + 128 + 32 + 32) = 256 channels
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        # 256 -> (128 + 192 + 96 + 64) = 480 channels
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 4: Five Inception modules
        # 480 -> (192 + 208 + 48 + 64) = 512 channels
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        # 512 -> (160 + 224 + 64 + 64) = 512 channels
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        # 512 -> (128 + 256 + 64 + 64) = 512 channels
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        # 512 -> (112 + 288 + 64 + 64) = 528 channels
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        # 528 -> (256 + 320 + 128 + 128) = 832 channels
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Global average pooling: flexible input size -> 1x1 output
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout layer with probability 0.4
        self.dropout = nn.Dropout(0.4)

        # Fully connected layer: 832 channels -> num_classes
        # After adaptive pooling: (batch, 832, 1, 1) -> flatten to 832
        self.fc = nn.Linear(832, num_classes)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Initial layers
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool2(x)

        # Stage 3: Inception modules
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)

        # Stage 4: Inception modules
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool4(x)

        # Classification head
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
