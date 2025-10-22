"""
Utilities Module.

Public API for utility functions including configuration parsing,
optimizer/loss creation, and visualization tools.

Exported Classes:
    Config: YAML configuration file parser with dot notation access

Exported Functions:
    create_criterion: Create loss function from configuration
    create_optimizer: Create optimizer from configuration
    create_scheduler: Create learning rate scheduler from configuration
    plot_training_history: Visualize training and validation metrics

Usage:
    >>> from utils import Config, create_optimizer, plot_training_history
    >>> cfg = Config('configs/flowers17_vgg.yaml')
    >>> optimizer = create_optimizer(model.parameters(), cfg)
    >>> plot_training_history(history, 'training_curves.png')
"""

from .config_parser import Config
from .optimizer_factory import (create_criterion, create_optimizer,
                                create_scheduler)
from .seed import set_seed
from .visualization import plot_training_history


__all__ = [
    'Config',
    'create_criterion',
    'create_optimizer',
    'create_scheduler',
    'plot_training_history',
    'set_seed'
]
