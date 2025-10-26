"""
VGG16 with Logistic Regression Classifier.

Uses pretrained VGG16 convolutional features (frozen) with a single linear
layer for classification.

Architecture:
    VGG16 Features (frozen) -> Flatten -> Linear(25088, num_classes)

Classes:
    VGG16LogisticRegression: VGG16 feature extractor + logistic regression

Example:
    >>> from models.custom.vgg16_logistic import VGG16LogisticRegression
    >>> model = VGG16LogisticRegression(num_classes=3)
    >>> output = model(torch.randn(1, 3, 224, 224))
"""

from torch import nn

from torchvision import models


class VGG16LogisticRegression(nn.Module):
    """VGG16 feature extractor with logistic regression classifier."""

    def __init__(self, num_classes):
        """
        Initialize VGG16 with frozen features and linear classifier.

        Args:
            num_classes: Number of output classes
        """
        super().__init__()

        # Load pretrained VGG16 and extract features only
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = vgg16.features

        # Freeze all feature parameters
        for param in self.features.parameters():
            param.requires_grad = False

        # Single linear layer (logistic regression)
        # VGG16 features output: 512 channels * 7 * 7 = 25088
        self.classifier = nn.Linear(512 * 7 * 7, num_classes)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
