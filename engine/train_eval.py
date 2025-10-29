"""
Training and Evaluation Engine.

Core training and evaluation loops for image classification models. Provides
efficient implementations for single-epoch training and model evaluation with
loss and top-1/top-5 accuracy metrics using torchmetrics.

Functions:
    train_one_epoch: Execute one training epoch with backpropagation
    evaluate: Evaluate model on validation or test set with optional
        prediction collection

Key Features:
    - Automatic device placement (CPU/CUDA/MPS)
    - Batch-wise loss accumulation for accurate epoch metrics
    - Memory-efficient evaluation with torch.no_grad()
    - Top-1 and Top-5 accuracy computation using torchmetrics
    - Optional prediction collection for detailed analysis

Example:
    >>> from engine import train_one_epoch, evaluate
    >>> train_loss, train_top1, train_top5 = train_one_epoch(
    ...     model, train_loader, optimizer, criterion, device, num_classes)
    >>> val_loss, val_top1, val_top5, preds, labels = evaluate(
    ...     model, val_loader, criterion, device, num_classes)
"""

import torch
from torchmetrics.classification import MulticlassAccuracy


def train_one_epoch(model, loader, optimizer, criterion, device, num_classes):
    """
    Execute one training epoch with backpropagation.

    Args:
        model: Neural network model to train
        loader: DataLoader providing training batches
        optimizer: Optimizer for updating model parameters
        criterion: Loss function
        device: Device to run training on ('cpu', 'cuda', or 'mps')
        num_classes: Number of classes in the dataset

    Returns:
        tuple: (average_loss, top1_accuracy, top5_accuracy) where:
            - average_loss (float): Mean loss across all samples
            - top1_accuracy (float): Top-1 accuracy as percentage (0-100)
            - top5_accuracy (float): Top-5 accuracy as percentage (0-100),
                or None if num_classes < 5
    """
    model.train()

    # Initialize metrics
    top1_metric = MulticlassAccuracy(num_classes=num_classes, top_k=1).to(device)
    top5_metric = MulticlassAccuracy(num_classes=num_classes, top_k=5).to(device) if num_classes >= 5 else None

    total_loss, total = 0.0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total += labels.size(0)

        # Update metrics
        top1_metric.update(outputs, labels)
        if top5_metric is not None:
            top5_metric.update(outputs, labels)

    avg_loss = total_loss / total
    top1_acc = top1_metric.compute().item() * 100.0
    top5_acc = top5_metric.compute().item() * 100.0 if top5_metric is not None else None

    return avg_loss, top1_acc, top5_acc

# pylint: disable=too-many-locals
def evaluate(model, loader, criterion, device, num_classes):
    """
    Evaluate model on validation or test set without gradient computation.

    This function computes loss and accuracy metrics while also collecting
    all predictions and true labels. The predictions can be used for
    generating detailed classification reports.

    Args:
        model: Neural network model to evaluate
        loader: DataLoader providing evaluation batches
        criterion: Loss function
        device: Device to run evaluation on ('cpu', 'cuda', or 'mps')
        num_classes: Number of classes in the dataset

    Returns:
        tuple: (average_loss, top1_accuracy, top5_accuracy, predictions, labels) where:
            - average_loss (float): Mean loss across all samples
            - top1_accuracy (float): Top-1 accuracy as percentage (0-100)
            - top5_accuracy (float): Top-5 accuracy as percentage (0-100),
                or None if num_classes < 5
            - predictions (list): List of predicted class indices
            - labels (list): List of true class indices

    Example:
        >>> # During training - get all metrics
        >>> val_loss, val_top1, val_top5, _, _ = evaluate(
        ...     model, val_loader, criterion, device, num_classes)
        >>> # Final evaluation - need predictions for report
        >>> test_loss, test_top1, test_top5, preds, labels = evaluate(
        ...     model, test_loader, criterion, device, num_classes)
    """
    model.eval()

    # Initialize metrics
    top1_metric = MulticlassAccuracy(num_classes=num_classes, top_k=1).to(device)
    top5_metric = MulticlassAccuracy(num_classes=num_classes, top_k=5).to(device) if num_classes >= 5 else None

    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(1)

            total_loss += loss.item() * imgs.size(0)
            total += labels.size(0)

            # Update metrics
            top1_metric.update(outputs, labels)
            if top5_metric is not None:
                top5_metric.update(outputs, labels)

            # Collect predictions and labels
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total
    top1_acc = top1_metric.compute().item() * 100.0
    top5_acc = top5_metric.compute().item() * 100.0 if top5_metric is not None else None

    return avg_loss, top1_acc, top5_acc, all_preds, all_labels
