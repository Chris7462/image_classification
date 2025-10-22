"""
Training Visualization Module.

Utilities for visualizing training progress and model performance. Generates
publication-quality plots of training and validation metrics over epochs.

Functions:
    plot_training_history: Create dual-plot figure showing loss and accuracy
        curves

Features:
    - Side-by-side loss and accuracy plots
    - Separate curves for training and validation sets
    - High-resolution output (300 DPI) suitable for reports
    - ggplot style for professional appearance
    - Markers for better readability

Example:
    >>> from utils import plot_training_history
    >>> history = {
    ...     'loss': [0.5, 0.4, 0.3],
    ...     'accuracy': [80, 85, 90],
    ...     'val_loss': [0.6, 0.5, 0.4],
    ...     'val_accuracy': [75, 80, 85]
    ... }
    >>> plot_training_history(history, 'outputs/plots/training_curves.png')
"""

import matplotlib.pyplot as plt

import numpy as np


def plot_training_history(history, save_path):
    """
    Plot training and validation loss and accuracy curves.

    Args:
        history: Dictionary with keys 'loss', 'accuracy', 'val_loss',
            'val_accuracy'
        save_path: Path to save the plot
    """
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 5))

    num_epochs = len(history['loss'])
    epochs_range = np.arange(1, num_epochs + 1)

    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['loss'], label='train_loss',
             marker='o', markersize=3)
    plt.plot(epochs_range, history['val_loss'], label='val_loss',
             marker='s', markersize=3)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['accuracy'], label='train_acc',
             marker='o', markersize=3)
    plt.plot(epochs_range, history['val_accuracy'], label='val_acc',
             marker='s', markersize=3)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Training curves saved to: {save_path}')
