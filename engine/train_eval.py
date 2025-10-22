"""
Training and Evaluation Engine.

Core training and evaluation loops for image classification models. Provides
efficient implementations for single-epoch training and model evaluation with
loss and accuracy metrics.

Functions:
    train_one_epoch: Execute one training epoch with backpropagation
    evaluate: Evaluate model on validation or test set with optional
        prediction collection

Key Features:
    - Automatic device placement (CPU/CUDA/MPS)
    - Batch-wise loss accumulation for accurate epoch metrics
    - Memory-efficient evaluation with torch.no_grad()
    - Top-1 accuracy computation
    - Optional prediction collection for detailed analysis

Example:
    >>> from engine import train_one_epoch, evaluate
    >>> train_loss, train_acc = train_one_epoch(model, train_loader, optimizer,
    ...     criterion, device)
    >>> val_loss, val_acc = evaluate(model, val_loader, criterion, device)[:2]
    >>> # For detailed evaluation with predictions:
    >>> test_loss, test_acc, preds, labels = evaluate(model, test_loader,
    ...     criterion, device)
"""

import torch


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Execute one training epoch with backpropagation.

    Args:
        model: Neural network model to train
        loader: DataLoader providing training batches
        optimizer: Optimizer for updating model parameters
        criterion: Loss function
        device: Device to run training on ('cpu', 'cuda', or 'mps')

    Returns:
        tuple: (average_loss, accuracy) where:
            - average_loss (float): Mean loss across all samples
            - accuracy (float): Top-1 accuracy as percentage (0-100)
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


# pylint: disable=too-many-locals
def evaluate(model, loader, criterion, device):
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

    Returns:
        tuple: (average_loss, accuracy, predictions, labels) where:
            - average_loss (float): Mean loss across all samples
            - accuracy (float): Top-1 accuracy as percentage (0-100)
            - predictions (list): List of predicted class indices
            - labels (list): List of true class indices

    Note:
        For training loop validation, you can unpack only the first two values:
            val_loss, val_acc = evaluate(...)[:2]
        For final test evaluation, unpack all four values for detailed reports.

    Example:
        >>> # During training - only need metrics
        >>> val_loss, val_acc = evaluate(model, val_loader, criterion,
        ...     device)[:2]
        >>> # Final evaluation - need predictions for report
        >>> test_loss, test_acc, preds, labels = evaluate(model, test_loader,
        ...     criterion, device)
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(1)

            total_loss += loss.item() * imgs.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Collect predictions and labels
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy, all_preds, all_labels
