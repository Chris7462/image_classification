"""
Configuration Parser Module

Simple YAML configuration file parser that converts nested dictionaries into
objects with dot notation attribute access. Provides clean and readable access
to configuration parameters throughout the codebase.

Classes:
    _ConfigNode: Internal class for nested config sections with attribute access
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

class _ConfigNode:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                v = _ConfigNode(v)
            setattr(self, k, v)

class Config:
    def __init__(self, yaml_file):
        import yaml
        with open(yaml_file, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        self.__dict__.update({k: _ConfigNode(v) if isinstance(v, dict) else v
                              for k, v in cfg_dict.items()})
