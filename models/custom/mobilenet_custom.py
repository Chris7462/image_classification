"""
MobileNet V1 Custom Architecture.

PyTorch implementation of MobileNetV1 for image classification, based on
"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications" (Howard et al., 2017). Features depthwise separable convolutions
for efficient computation with width multiplier 1.0.

Architecture:
    Conv3x3(32) -> 13x DepthwiseSeparable blocks -> AvgPool -> FC(num_classes)

    DepthwiseSeparable blocks configuration:
    - (64, 1):   64 filters, stride 1
    - (128, 2):  128 filters, stride 2 (downsample)
    - (128, 1):  128 filters, stride 1
    - (256, 2):  256 filters, stride 2 (downsample)
    - (256, 1):  256 filters, stride 1
    - (512, 2):  512 filters, stride 2 (downsample)
    - (512, 1):  512 filters, stride 1 (repeated 5 times)
    - (1024, 2): 1024 filters, stride 2 (downsample)
    - (1024, 1): 1024 filters, stride 1

Key Features:
    - Depthwise separable convolutions (depthwise + pointwise)
    - Batch normalization for training stability
    - Width multiplier 1.0 (standard model)
    - ReLU6 activation for mobile deployment
    - Designed for 224x224 RGB input images

Classes:
    DepthwiseSeparableConv: Depthwise separable convolution block
    MobileNetCustom: Complete MobileNetV1 architecture

Example:
    >>> from models.custom.mobilenet_custom import MobileNetCustom
    >>> model = MobileNetCustom(num_classes=1000)
    >>> output = model(torch.randn(1, 3, 224, 224))
"""

import torch
from torch import nn


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution block.

    Consists of:
    1. Depthwise convolution: applies a single filter per input channel
    2. Pointwise convolution: 1x1 convolution to combine channels

    This factorization reduces parameters and computation compared to
    standard convolutions.

    Architecture:
        Depthwise Conv3x3 -> BN -> ReLU6 -> Pointwise Conv1x1 -> BN -> ReLU6
    """

    def __init__(self, in_channels, out_channels, stride):
        """
        Initialize depthwise separable convolution block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for depthwise convolution (1 or 2)
        """
        super().__init__()

        # Depthwise convolution: one filter per input channel
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                     stride=stride, padding=1, groups=in_channels,
                     bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )

        # Pointwise convolution: 1x1 conv to change channels
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass through depthwise separable convolution.

        Args:
            x: Input tensor

        Returns:
            Output tensor after depthwise and pointwise convolutions
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileNetCustom(nn.Module):
    """
    MobileNetV1 architecture with width multiplier 1.0.

    Implements the MobileNetV1 architecture using depthwise separable
    convolutions for efficient mobile and embedded vision applications.
    """

    def __init__(self, num_classes):
        """
        Initialize MobileNetV1 for 224x224 RGB images.

        Args:
            num_classes: Number of output classes
        """
        super().__init__()

        # Initial standard convolution: 224x224x3 -> 112x112x32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        # MobileNet body: depthwise separable convolution blocks
        # Configuration: (out_channels, stride)
        self.features = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=1),    # 112x112x32 -> 112x112x64
            DepthwiseSeparableConv(64, 128, stride=2),   # 112x112x64 -> 56x56x128
            DepthwiseSeparableConv(128, 128, stride=1),  # 56x56x128 -> 56x56x128
            DepthwiseSeparableConv(128, 256, stride=2),  # 56x56x128 -> 28x28x256
            DepthwiseSeparableConv(256, 256, stride=1),  # 28x28x256 -> 28x28x256
            DepthwiseSeparableConv(256, 512, stride=2),  # 28x28x256 -> 14x14x512
            DepthwiseSeparableConv(512, 512, stride=1),  # 14x14x512 -> 14x14x512
            DepthwiseSeparableConv(512, 512, stride=1),  # 14x14x512 -> 14x14x512
            DepthwiseSeparableConv(512, 512, stride=1),  # 14x14x512 -> 14x14x512
            DepthwiseSeparableConv(512, 512, stride=1),  # 14x14x512 -> 14x14x512
            DepthwiseSeparableConv(512, 512, stride=1),  # 14x14x512 -> 14x14x512
            DepthwiseSeparableConv(512, 1024, stride=2), # 14x14x512 -> 7x7x1024
            DepthwiseSeparableConv(1024, 1024, stride=1) # 7x7x1024 -> 7x7x1024
        )

        # Global average pooling: 7x7x1024 -> 1x1x1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected classifier: 1024 -> num_classes
        self.fc = nn.Linear(1024, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize model weights using best practices.

        Applies:
        - Kaiming normal initialization for Conv2d layers (optimal for ReLU)
        - Constant initialization for BatchNorm2d (weight=1, bias=0)
        - Normal initialization for Linear layers (mean=0, std=0.01)
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
        x = self.conv1(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
