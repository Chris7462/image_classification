"""
Models Module.

Public API for model creation and architecture definitions. Provides factory
functions for instantiating various neural network architectures from
configuration.

Exported Functions:
    create_model: Create and initialize model from configuration

Usage:
    >>> from models import create_model
    >>> model = create_model(cfg)
"""

from .model_factory import create_model

__all__ = [
    'create_model'
]
