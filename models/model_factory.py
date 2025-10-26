"""
Model Factory Module.

Factory pattern implementation for creating image classification models from
configuration files. Supports custom architectures and pretrained models with
optional backbone freezing for transfer learning.

Supported Models:
    - minivggnet: Custom lightweight CNN for small images
    - vgg16: VGG16 with ImageNet pretrained weights
    - vgg16_logistic: VGG16 features + logistic regression
    - resnet50: ResNet50 with ImageNet pretrained weights

Transfer Learning Features:
    - Load pretrained weights from torchvision
    - Freeze backbone layers while training only the classifier
    - Automatic classifier head replacement for custom number of classes

Functions:
    create_model: Instantiate model from config with proper initialization

Example:
    >>> from utils import Config
    >>> from models import create_model
    >>> cfg = Config('configs/flowers17_vgg.yaml')
    >>> model = create_model(cfg)
"""

from models.custom.minivggnet import MiniVGGNet
from models.custom.vgg16_logistic import VGG16LogisticRegression

from torch import nn

from torchvision import models


def create_model(cfg):
    """
    Instantiate model from configuration with proper initialization.

    Creates an image classification model based on the backbone specified
    in the configuration. Supports custom architectures and pretrained
    models with optional backbone freezing for transfer learning.

    Args:
        cfg: Configuration object with model section containing:
            - backbone (str): Model architecture name
            - num_classes (int): Number of output classes
            - pretrained (bool): Use pretrained weights (for vgg16/resnet50)
            - freeze_backbone (bool): Freeze backbone layers (for transfer
              learning)

    Returns:
        torch.nn.Module: Instantiated model ready for training

    Raises:
        ValueError: If backbone is not supported

    Example:
        >>> cfg = Config('configs/flowers17_vgg.yaml')
        >>> model = create_model(cfg)
        >>> print(model)
    """
    backbone = cfg.model.backbone.lower()
    num_classes = cfg.model.num_classes

    if backbone == 'minivggnet':
        # Custom MiniVGGNet doesn't use pretrained weights
        model = MiniVGGNet(num_classes=num_classes)

    elif backbone == 'vgg16_logistic':
        # VGG16 with frozen features and logistic regression classifier
        model = VGG16LogisticRegression(num_classes=num_classes)

    elif backbone == 'vgg16':
        pretrained = cfg.model.pretrained
        freeze_backbone = cfg.model.freeze_backbone
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1
                             if pretrained else None)
        if freeze_backbone:
            for p in model.features.parameters():
                p.requires_grad = False
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif backbone == 'resnet50':
        pretrained = cfg.model.pretrained
        freeze_backbone = cfg.model.freeze_backbone
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1
                                if pretrained else None)
        if freeze_backbone:
            for p in list(model.children())[:-1]:
                for param in p.parameters():
                    param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f'Unsupported backbone: {backbone}')

    return model
