"""
Resnet50 with Logistic Regression Classifier.

Uses pretrained Resnet50 convolutional features (frozen) with a single linear
layer for classification.

Architecture:
    Resnet50 Features (frozen) -> Flatten -> Linear(2048, num_classes)

Classes:
    Resnet50LogisticRegression: Resnet50 feature extractor + logistic regression

Example:
    >>> from models.custom.resnet50_logistic import Resnet50LogisticRegression
    >>> model = Resnet50LogisticRegression(num_classes=3)
    >>> output = model(torch.randn(1, 3, 224, 224))
"""

from torch import nn

from torchvision import models


class Resnet50LogisticRegression(nn.Module):
    """Resnet50 feature extractor with logistic regression classifier."""

    def __init__(self, num_classes):
        """
        Initialize Resnet50 with frozen features and linear classifier.

        Args:
            num_classes: Number of output classes
        """
        super().__init__()

        # Load pretrained VGG16 and extract features only
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Extract all layers except the last fully connected layer
        modules = list(resnet50.children())[:-1]
        self.features = nn.Sequential(*modules)

        # Freeze all feature parameters
        for param in self.features.parameters():
            param.requires_grad = False

        # Single linear layer (logistic regression)
        # ResNet50 features output: 2048 channels * 1 * 1 = 2048
        self.classifier = nn.Linear(2048, num_classes)

        #   # Check frozen status
        #   for name, param in features.named_parameters():
        #       print(f"{name}: requires_grad = {param.requires_grad}")

        #   # Count trainable parameters
        #   trainable = sum(p.numel() for p in features.parameters() if p.requires_grad)
        #   print(f"Trainable parameters: {trainable}")  #

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
        return x

#   from torchvision import models
#   from torchinfo import summary

#   resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
#   summary(resnet50, input_size=(1, 3, 224, 224), device='cpu')
