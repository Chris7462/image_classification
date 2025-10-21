from .custom.minivggnet import MiniVGGNet
from .model_factory import create_model

MODEL_REGISTRY = {
    "MiniVGGNet": MiniVGGNet,
}

__all__ = ['MiniVGGNet', 'MODEL_REGISTRY', 'create_model']
