"""
Training Engine Module.

Public API for training and evaluation loops. Provides the core functionality
for model training and performance evaluation with top-1 and top-5 accuracy.

Exported Functions:
    train_one_epoch: Execute one training epoch with backpropagation
    evaluate: Evaluate model on validation or test set
    evaluate_with_multicrop: Evaluate with FiveCrop or TenCrop augmentation

Usage:
    >>> from engine import train_one_epoch, evaluate, evaluate_with_multicrop
    >>> train_loss, train_top1, train_top5 = train_one_epoch(
    ...     model, loader, optimizer, criterion, device, num_classes)
    >>> val_loss, val_top1, val_top5, preds, labels = evaluate(
    ...     model, loader, criterion, device, num_classes)
    >>> # FiveCrop or TenCrop evaluation
    >>> loss, top1, top5, preds, labels = evaluate_with_multicrop(
    ...     model, loader, criterion, device, num_classes, num_crops=5)
"""

# Import main classes and functions
from .train_eval import evaluate, evaluate_with_multicrop, train_one_epoch

__all__ = [
    'evaluate',
    'evaluate_with_multicrop',
    'train_one_epoch'
]
