"""
Training and Evaluation Engine

Core training and evaluation loops for image classification models. Provides
efficient implementations for single-epoch training and model evaluation with
loss and accuracy metrics.

Functions:
    train_one_epoch: Execute one training epoch with backpropagation
    evaluate: Evaluate model on validation or test set without gradient computation

Key Features:
    - Automatic device placement (CPU/CUDA/MPS)
    - Batch-wise loss accumulation for accurate epoch metrics
    - Memory-efficient evaluation with torch.no_grad()
    - Top-1 accuracy computation

Example:
    >>> from engine import train_one_epoch, evaluate
    >>> train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    >>> val_loss, val_acc = evaluate(model, val_loader, criterion, device)
"""

import torch


def train_one_epoch(model, loader, optimizer, criterion, device):
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

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, 100.0 * correct / total
