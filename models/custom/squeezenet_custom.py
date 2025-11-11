"""
SqueezeNet Architecture.

PyTorch implementation of SqueezeNet 1.0 for image classification, based on
"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model
size" (Iandola et al., 2016). Features Fire modules with squeeze and expand
layers for efficient computation.

Architecture:
    Conv7x7(96) -> MaxPool -> Fire(16,64,64) -> Fire(16,64,64) ->
    Fire(32,128,128) -> MaxPool -> Fire(32,128,128) -> Fire(48,192,192) ->
    Fire(48,192,192) -> Fire(64,256,256) -> MaxPool -> Fire(64,256,256) ->
    Dropout(0.5) -> Conv1x1(num_classes) -> ReLU -> AdaptiveAvgPool

Key Features:
    - Fire modules: squeeze layer (1x1) followed by expand layers (1x1 + 3x3)
    - AlexNet-level accuracy with 50x fewer parameters
    - Compact model size (~5MB)
    - Designed for 224x224 RGB input images
    - No batch normalization (original paper design)

Classes:
    Fire: Fire module with squeeze and expand layers
    SqueezeNet: Complete SqueezeNet 1.0 architecture

Example:
    >>> from models.custom.squeezenet import SqueezeNet
    >>> model = SqueezeNet(num_classes=1000)
    >>> output = model(torch.randn(1, 3, 224, 224))
"""

import torch
import torch.nn as nn
import torch.nn.init as init


class Fire(nn.Module):
    """
    Fire module for SqueezeNet.

    A Fire module consists of a squeeze convolution layer (1x1 filters),
    feeding into an expand layer that has a mix of 1x1 and 3x3 convolution
    filters. This design reduces parameters while maintaining accuracy.

    Architecture:
        Input -> Conv1x1(squeeze) -> ReLU ->
            ├─> Conv1x1(expand1x1) -> ReLU ─┐
            └─> Conv3x3(expand3x3) -> ReLU ─┴─> Concatenate -> Output
    """

    def __init__(self, in_channels, squeeze_channels, expand1x1_channels,
                 expand3x3_channels):
        """
        Initialize Fire module.

        Args:
            in_channels: Number of input channels
            squeeze_channels: Number of filters in squeeze layer (1x1 conv)
            expand1x1_channels: Number of filters in 1x1 expand layer
            expand3x3_channels: Number of filters in 3x3 expand layer

        Note:
            Output channels = expand1x1_channels + expand3x3_channels
        """
        super(Fire, self).__init__()

        # Squeeze layer: 1x1 convolution to reduce channels
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)

        # Expand layer: parallel 1x1 and 3x3 convolutions
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)

        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass through Fire module.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Output tensor of shape (batch_size, expand1x1_channels +
                expand3x3_channels, height, width)
        """
        # Squeeze phase
        x = self.squeeze_activation(self.squeeze(x))

        # Expand phase: concatenate 1x1 and 3x3 outputs
        out1 = self.expand1x1_activation(self.expand1x1(x))
        out2 = self.expand3x3_activation(self.expand3x3(x))

        return torch.cat([out1, out2], dim=1)


class SqueezeNetCustom(nn.Module):
    """
    SqueezeNet 1.0 architecture for image classification.

    Implements the original SqueezeNet architecture with Fire modules for
    efficient computation. Achieves AlexNet-level accuracy with 50x fewer
    parameters (~1.2M parameters vs ~60M for AlexNet).

    Note:
        Designed for 224x224 RGB input images. The architecture uses no
        batch normalization, following the original paper design.
    """

    def __init__(self, num_classes):
        """
        Initialize SqueezeNet 1.0 for 224x224 RGB images.

        Args:
            num_classes: Number of output classes (default: 1000 for ImageNet)
        """
        super().__init__()
        self.num_classes = num_classes

        # Feature extraction layers
        self.features = nn.Sequential(
            # Initial convolution: 224x224x3 -> 111x111x96
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            # 111x111x96 -> 55x55x96
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            # Fire modules: 55x55x96 -> 55x55x128
            Fire(96, 16, 64, 64),     # 96 -> 128
            Fire(128, 16, 64, 64),    # 128 -> 128
            Fire(128, 32, 128, 128),  # 128 -> 256
            # 55x55x256 -> 27x27x256
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            # Fire modules: 27x27x256 -> 27x27x512
            Fire(256, 32, 128, 128),  # 256 -> 256
            Fire(256, 48, 192, 192),  # 256 -> 384
            Fire(384, 48, 192, 192),  # 384 -> 384
            Fire(384, 64, 256, 256),  # 384 -> 512
            # 27x27x512 -> 13x13x512
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            # Fire module: 13x13x512 -> 13x13x512
            Fire(512, 64, 256, 256),  # 512 -> 512
        )

        # Classifier
        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    # Final classifier conv: normal initialization
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    # All other convs: Kaiming uniform initialization
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)
