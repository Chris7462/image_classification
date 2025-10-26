"""
Optimizer and Loss Factory Module.

Factory functions for creating loss functions, optimizers, and learning rate
schedulers from configuration files. Supports common optimization algorithms
and scheduling strategies for deep learning.

Supported Components:
    Loss Functions: CrossEntropy, MSE, BCE, BCEWithLogits
    Optimizers: Adam, SGD, AdamW, RMSprop
    Schedulers: StepLR, CosineAnnealing, ReduceLROnPlateau

Functions:
    create_criterion: Create loss function from config
    create_optimizer: Create optimizer with hyperparameters from config
    create_scheduler: Create learning rate scheduler from config

Example:
    >>> from utils import Config, create_criterion, create_optimizer \
    ...     create_scheduler
    >>> cfg = Config('configs/flowers17_vgg.yaml')
    >>> criterion = create_criterion(cfg)
    >>> optimizer = create_optimizer(model.parameters(), cfg)
    >>> scheduler = create_scheduler(optimizer, cfg)
"""

from torch import nn, optim


def create_criterion(cfg):
    """
    Create loss criterion from configuration.

    Args:
        cfg: Configuration object with loss.type attribute

    Returns:
        torch.nn.Module: Loss function

    Raises:
        ValueError: If loss type is not supported
    """
    loss_type = cfg.loss.type.lower()

    if loss_type == 'cross_entropy':    # pylint: disable=no-else-return
        return nn.CrossEntropyLoss()
    elif loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'bce':
        return nn.BCELoss()
    elif loss_type == 'bce_with_logits':
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f'Unsupported loss type: {loss_type}')


def create_optimizer(model_parameters, cfg, weight_decay=None):
    """
    Create optimizer from configuration.

    Args:
        model_parameters: Model parameters to optimize
        cfg: Configuration object with optimizer settings
        weight_decay: Optional weight_decay override (if None, uses config
            value)

    Returns:
        torch.optim.Optimizer: Configured optimizer

    Raises:
        ValueError: If optimizer type is not supported
    """
    opt_type = cfg.optimizer.type.lower()
    lr = cfg.optimizer.lr

    # Use provided weight_decay or fall back to config
    if weight_decay is None:
        weight_decay = cfg.optimizer.weight_decay

    if opt_type == 'adam':  # pylint: disable=no-else-return
        return optim.Adam(model_parameters, lr=lr,
                          weight_decay=weight_decay)
    elif opt_type == 'sgd':
        momentum = (cfg.optimizer.momentum if hasattr(cfg.optimizer,
                                                      'momentum') else 0.9)
        return optim.SGD(model_parameters, lr=lr,
                         weight_decay=weight_decay, momentum=momentum)
    elif opt_type == 'adamw':
        return optim.AdamW(model_parameters, lr=lr,
                           weight_decay=weight_decay)
    elif opt_type == 'rmsprop':
        return optim.RMSprop(model_parameters, lr=lr,
                             weight_decay=weight_decay)
    else:
        raise ValueError(f'Unsupported optimizer type: {opt_type}')


def create_scheduler(optimizer, cfg):
    """
    Create learning rate scheduler from configuration.

    Args:
        optimizer: Optimizer to schedule
        cfg: Configuration object with scheduler settings

    Returns:
        torch.optim.lr_scheduler: Learning rate scheduler or None

    Raises:
        ValueError: If scheduler type is not supported
    """
    if not hasattr(cfg, 'scheduler') or cfg.scheduler.type.lower() == 'none':
        return None

    sched_type = cfg.scheduler.type.lower()

    if sched_type == 'step_lr':  # pylint: disable=no-else-return
        return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=cfg.scheduler.step_size,
                gamma=cfg.scheduler.gamma
                )
    elif sched_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.scheduler.t_max
                )
    elif sched_type == 'reduce_on_plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=cfg.scheduler.factor,
                patience=cfg.scheduler.patience
                )
    else:
        raise ValueError(f'Unsupported scheduler type: {sched_type}')
