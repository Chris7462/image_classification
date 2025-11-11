"""
GoogLeNet (Inception v1) Architecture.

PyTorch implementation of GoogLeNet from "Going Deeper with Convolutions"
(Szegedy et al., 2014). Features full 4-branch Inception modules with parallel
1x1, 3x3, and 5x5 convolutions plus max pooling branches for multi-scale
feature extraction.

Architecture:
    Initial layers: Conv7x7 -> MaxPool -> Conv1x1 -> Conv3x3 -> MaxPool
    Stage 3: 2xInception -> MaxPool
    Stage 4: 5xInception -> MaxPool
    Stage 5: 2xInception -> AvgPool
    Classifier: Dropout(0.4) -> FC(num_classes)

Key Features:
    - Full 4-branch Inception modules (1x1, 1x1→3x3, 1x1→5x5, MaxPool→1x1)
    - Dimensionality reduction with 1x1 convolutions before expensive operations
    - Batch normalization with eps=0.001 for training stability
    - Kaiming initialization for Conv layers
    - Adaptive pooling for flexible input image sizes
    - Dropout regularization (0.4) to prevent overfitting
    - Designed for 224x224 RGB input images

Classes:
    ConvModule: Basic convolution block with BN and ReLU
    Inception: Full 4-branch parallel convolution module
    GoogLeNetCustom: Complete architecture

Example:
    >>> from models.custom.googlenet import GoogLeNetCustom
    >>> model = GoogLeNetCustom(num_classes=1000)
    >>> output = model(torch.randn(1, 3, 224, 224))
"""

import torch
import torch.nn as nn


class ConvModule(nn.Module):
    """
    Basic convolution block: Conv2d -> BatchNorm2d -> ReLU.

    This module provides a reusable building block for the network with
    proper normalization and activation. Uses bias=False since BatchNorm
    makes the bias term redundant.
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
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
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


class Inception(nn.Module):
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

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5,
                 pool_proj):
        """
        Initialize Inception module with four branches.

        Args:
            in_channels: Number of input channels
            ch1x1: Number of filters in 1x1 convolution branch
            ch3x3red: Number of filters in 1x1 reduction before 3x3 conv
            ch3x3: Number of filters in 3x3 convolution
            ch5x5red: Number of filters in 1x1 reduction before 5x5 conv
            ch5x5: Number of filters in 5x5 convolution
            pool_proj: Number of filters in 1x1 projection after max pooling
        """
        super().__init__()

        # Branch 1: 1x1 conv
        self.branch1 = ConvModule(in_channels, ch1x1, kernel_size=1,
                                  stride=1, padding=0)

        # Branch 2: 1x1 conv -> 3x3 conv
        self.branch2 = nn.Sequential(
            ConvModule(in_channels, ch3x3red, kernel_size=1,
                      stride=1, padding=0),
            ConvModule(ch3x3red, ch3x3, kernel_size=3,
                      stride=1, padding=1)
        )

        # Branch 3: 1x1 conv -> 5x5 conv
        self.branch3 = nn.Sequential(
            ConvModule(in_channels, ch5x5red, kernel_size=1,
                      stride=1, padding=0),
            ConvModule(ch5x5red, ch5x5, kernel_size=5,
                      stride=1, padding=2)
        )

        # Branch 4: 3x3 pool -> 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvModule(in_channels, pool_proj, kernel_size=1,
                      stride=1, padding=0)
        )

    def forward(self, x):
        """
        Forward pass through all four branches with concatenation.

        Args:
            x: Input tensor

        Returns:
            Concatenated output from all branches along channel dimension
        """
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class GoogLeNetCustom(nn.Module):
    """
    GoogLeNet (Inception v1) architecture for image classification.

    A deep convolutional network that uses Inception modules with parallel
    convolutions at multiple scales. Uses adaptive pooling to support
    flexible input image sizes (minimum 224x224 recommended).
    """

    def __init__(self, num_classes):
        """
        Initialize GoogLeNet for RGB images.

        Args:
            num_classes: Number of output classes (default: 1000 for ImageNet)
        """
        super(GoogLeNetCustom, self).__init__()

        # Initial convolutional layers
        self.pre_layers = nn.Sequential(
            ConvModule(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            ConvModule(64, 64, kernel_size=1, stride=1, padding=0),
            ConvModule(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2, ceil_mode=True)
        )

        # Stage 3: Two Inception modules
        # 192 -> (64 + 128 + 32 + 32) = 256 channels
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        # 256 -> (128 + 192 + 96 + 64) = 480 channels
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # Stage 4: Five Inception modules
        # 480 -> (192 + 208 + 48 + 64) = 512 channels
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        # 512 -> (160 + 224 + 64 + 64) = 512 channels
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        # 512 -> (128 + 256 + 64 + 64) = 512 channels
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        # 512 -> (112 + 288 + 64 + 64) = 528 channels
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        # 528 -> (256 + 320 + 128 + 128) = 832 channels
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # Stage 5: Two Inception modules
        # 832 -> (256 + 320 + 128 + 128) = 832 channels
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        # 832 -> (384 + 384 + 128 + 128) = 1024 channels
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        # Global average pooling: flexible input size -> 1x1 output
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout layer with probability 0.4
        self.dropout = nn.Dropout(0.4)

        # Fully connected layer: 1024 channels -> num_classes
        # After adaptive pooling: (batch, 1024, 1, 1) -> flatten to 1024
        self.linear = nn.Linear(1024, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize model weights using best practices.

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
            x: Input tensor of shape (batch_size, 3, height, width)
               Minimum recommended size: 224x224

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Initial layers
        x = self.pre_layers(x)

        # Stage 3: Inception modules
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool1(x)

        # Stage 4: Inception modules
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool2(x)

        # Stage 5: Inception modules
        x = self.inception5a(x)
        x = self.inception5b(x)

        # Classification head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.linear(x)

        return x
