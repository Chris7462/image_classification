"""
Configuration Parser Module.

Simple YAML configuration file parser that converts nested dictionaries into
objects with dot notation attribute access. Provides clean and readable access
to configuration parameters throughout the codebase.

Classes:
    _ConfigNode: Internal class for nested config sections with attribute
        access
    Config: Main configuration class that loads YAML and provides dot notation

Features:
    - Nested configuration support (e.g., cfg.dataset.batch_size)
    - Clean dot notation instead of dictionary keys
    - Automatic conversion of nested dicts to ConfigNode objects

Example:
    >>> from utils import Config
    >>> cfg = Config('configs/flowers17_vgg.yaml')
    >>> print(cfg.dataset.batch_size)  # Access with dot notation
    32
    >>> print(cfg.model.backbone)
    'vgg16'
"""

import yaml


class _ConfigNode:  # pylint: disable=too-few-public-methods
    """Internal class for nested configuration sections with dot notation."""

    def __init__(self, d):
        """
        Initialize ConfigNode from dictionary.

        Args:
            d: Dictionary to convert to object with attribute access
        """
        for k, v in d.items():
            if isinstance(v, dict):
                v = _ConfigNode(v)
            setattr(self, k, v)


class Config:  # pylint: disable=too-few-public-methods
    """
    Main configuration class that loads YAML files.

    Provides dot notation access to nested configuration parameters.
    Converts nested dictionaries into ConfigNode objects for clean access.

    Example:
        >>> cfg = Config('configs/my_config.yaml')
        >>> batch_size = cfg.dataset.batch_size
    """

    def __init__(self, yaml_file):
        """
        Load configuration from YAML file.

        Args:
            yaml_file: Path to YAML configuration file
        """
        with open(yaml_file, 'r', encoding='utf-8') as f:
            cfg_dict = yaml.safe_load(f)
        self.__dict__.update({k: _ConfigNode(v) if isinstance(v, dict) else v
                              for k, v in cfg_dict.items()})
