"""
ResNet Custom Architecture.

PyTorch implementation of ResNet (Residual Network) architectures for image
classification, based on "Deep Residual Learning for Image Recognition"
(He et al., 2015). Supports both BasicBlock (ResNet-18/34) and Bottleneck
(ResNet-50/101/152) designs for training from scratch.

Architecture variants:
    ResNet-18:  BasicBlock with [2, 2, 2, 2] blocks per stage (~11M parameters)
    ResNet-34:  BasicBlock with [3, 4, 6, 3] blocks per stage (~21M parameters)
    ResNet-50:  Bottleneck with [3, 4, 6, 3] blocks per stage (~25M parameters)
    ResNet-101: Bottleneck with [3, 4, 23, 3] blocks per stage (~44M parameters)
    ResNet-152: Bottleneck with [3, 8, 36, 3] blocks per stage (~60M parameters)

Key Features:
    - Skip connections for gradient flow in deep networks
    - Batch normalization for training stability
    - Both BasicBlock (2 conv layers) and Bottleneck (3 conv layers) designs
    - Adaptive pooling for flexible output dimensions
    - Designed for 224x224 input images (ImageNet-style)

Classes:
    BasicBlock: Two-layer residual block for ResNet-18/34
    Bottleneck: Three-layer bottleneck block for ResNet-50/101/152
    ResNetCustom: Complete ResNet architecture

Functions:
    resnet18_custom: Create ResNet-18 model
    resnet34_custom: Create ResNet-34 model
    resnet50_custom: Create ResNet-50 model
    resnet101_custom: Create ResNet-101 model
    resnet152_custom: Create ResNet-152 model

Example:
    >>> from models.custom.resnet_custom import resnet50_custom
    >>> model = resnet50_custom(num_classes=10)
    >>> output = model(torch.randn(1, 3, 224, 224))
"""

import torch
from torch import nn


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-18 and ResNet-34.

    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+shortcut) -> ReLU

    The expansion factor is 1, meaning the output channels equal the
    specified number of channels.
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Initialize BasicBlock.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for the first convolution (1 or 2)
            downsample: Optional downsample layer for the shortcut connection
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        """
        Forward pass through the basic block.

        Args:
            x: Input tensor

        Returns:
            Output tensor after residual connection and activation
        """
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck residual block for ResNet-50, ResNet-101, and ResNet-152.

    Structure: Conv1x1 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU ->
               Conv1x1 -> BN -> (+shortcut) -> ReLU

    The expansion factor is 4, meaning the final 1x1 convolution expands
    the channels by 4x (e.g., 64 -> 256).
    """

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Initialize Bottleneck block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of intermediate channels (final will be
                         out_channels * expansion)
            stride: Stride for the 3x3 convolution (1 or 2)
            downsample: Optional downsample layer for the shortcut connection
        """
        super(Bottleneck, self).__init__()
        width = out_channels

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        """
        Forward pass through the bottleneck block.

        Args:
            x: Input tensor

        Returns:
            Output tensor after residual connection and activation
        """
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetCustom(nn.Module):
    """
    ResNet architecture for image classification.

    Implements the standard ResNet architecture with configurable block type
    and layer depths. Designed for 224x224 input images with initial 7x7
    convolution and max pooling.
    """

    def __init__(self, block, layers, num_classes):
        """
        Initialize ResNet model.

        Args:
            block: Block type (BasicBlock or Bottleneck)
            layers: List of integers specifying number of blocks in each stage
                   (e.g., [3, 4, 6, 3] for ResNet-50)
            num_classes: Number of output classes
        """
        super().__init__()
        self.in_channels = 64

        # Initial convolution: 7x7, stride 2, padding 3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build residual stages
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        Create a residual stage with specified number of blocks.

        Args:
            block: Block type (BasicBlock or Bottleneck)
            out_channels: Number of output channels for this stage
            blocks: Number of residual blocks in this stage
            stride: Stride for the first block (used for downsampling)

        Returns:
            Sequential container of residual blocks
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride,
                           downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# Factory functions for different ResNet variants

def resnet18_custom(num_classes):
    """
    Create ResNet-18 model.

    Args:
        num_classes: Number of output classes

    Returns:
        ResNet-18 model with BasicBlock and [2, 2, 2, 2] configuration
    """
    return ResNetCustom(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34_custom(num_classes):
    """
    Create ResNet-34 model.

    Args:
        num_classes: Number of output classes

    Returns:
        ResNet-34 model with BasicBlock and [3, 4, 6, 3] configuration
    """
    return ResNetCustom(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50_custom(num_classes):
    """
    Create ResNet-50 model.

    Args:
        num_classes: Number of output classes

    Returns:
        ResNet-50 model with Bottleneck and [3, 4, 6, 3] configuration
    """
    return ResNetCustom(Bottleneck, [3, 4, 6, 3], num_classes)


def resnet101_custom(num_classes):
    """
    Create ResNet-101 model.

    Args:
        num_classes: Number of output classes

    Returns:
        ResNet-101 model with Bottleneck and [3, 4, 23, 3] configuration
    """
    return ResNetCustom(Bottleneck, [3, 4, 23, 3], num_classes)


def resnet152_custom(num_classes):
    """
    Create ResNet-152 model.

    Args:
        num_classes: Number of output classes

    Returns:
        ResNet-152 model with Bottleneck and [3, 8, 36, 3] configuration
    """
    return ResNetCustom(Bottleneck, [3, 8, 36, 3], num_classes)
