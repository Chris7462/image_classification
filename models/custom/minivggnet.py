import torch.nn as nn


class MiniVGGNet(nn.Module):
    def __init__(self, num_classes):
        """
        MiniVGGNet with adaptive pooling for flexible input sizes.

        Args:
            num_classes: Number of output classes
        """
        super(MiniVGGNet, self).__init__()

        # Block 1: CONV => RELU => CONV => RELU => POOL, with BatchNorm and Dropout.
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25)
        )

        # Block 2: CONV => RELU => CONV => RELU => POOL, with BatchNorm and Dropout.
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
        )

        # Adaptive pooling to fixed spatial size regardless of input dimensions
        # This outputs 64 channels * 7 * 7 spatial dimensions = 3136 features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Calculate flattened dimension: 64 channels * 7 * 7 = 3136
        flattened_dim = 64 * 7 * 7

        # Fully connected layers: Flatten, Dense, ReLU, BatchNorm, Dropout, Dense.
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flattened_dim, out_features=512),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),
            nn.Dropout(p=0.5),
            # logit for softmax classifier
            nn.Linear(in_features=512, out_features=num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.adaptive_pool(x)
        x = self.fc(x)
        return x
