"""
Training Engine Module.

Public API for training and evaluation loops. Provides the core functionality
for model training and performance evaluation.

Exported Functions:
    train_one_epoch: Execute one training epoch with backpropagation
    evaluate: Evaluate model on validation or test set

Usage:
    >>> from engine import train_one_epoch, evaluate
    >>> train_loss, train_acc = train_one_epoch(model, loader, optimizer,
    ...     criterion, device)
    >>> val_loss, val_acc = evaluate(model, loader, criterion, device)
"""

# Import main classes and functions
from .train_eval import evaluate, train_one_epoch

__all__ = [
    'evaluate',
    'train_one_epoch'
]
