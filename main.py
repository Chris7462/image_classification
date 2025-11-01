"""
Image Classification Training Script.

Main entry point for training image classification models. Orchestrates the
entire training pipeline including data loading, model creation, training
loop, evaluation, and visualization. Supports FiveCrop and TenCrop test-time
augmentation for improved test accuracy.

Usage:
    python main.py --config configs/flowers17_vgg.yaml
    python main.py --config configs/flowers17_minivggnet.yaml
    python main.py --config configs/dogs_vs_cats_alexnet.yaml

Command-line Arguments:
    --config: Path to YAML configuration file (required)

Outputs:
    - outputs/checkpoints/<model_name>.pth: Best model weights
    - outputs/plots/<model_name>.png: Training/validation curves
"""


import argparse
import os

from data import get_data_loaders

from engine import evaluate, evaluate_with_multicrop, train_one_epoch

from models import create_model

from sklearn.metrics import classification_report

import torch

from utils import (Config, create_criterion, create_optimizer,
                   create_scheduler, plot_training_history, set_seed)


class Trainer:
    """Orchestrates the training pipeline for image classification."""

    def __init__(self, cfg):
        """
        Initialize trainer with configuration.

        Args:
            cfg: Configuration object from YAML file
        """
        self.cfg = cfg
        self.device = self._setup_device()
        self.checkpoint_path, self.plot_path = self._setup_directories()

        # Data loaders
        self.train_loader, self.val_loader, self.test_loader = \
            get_data_loaders(cfg)

        # Store class names and num_classes
        self.class_names = self.train_loader.dataset.classes
        self.num_classes = len(self.class_names)

        # Model components
        self.model = create_model(cfg).to(self.device)
        self.criterion = create_criterion(cfg)
        self.optimizer = create_optimizer(
            filter(lambda p: p.requires_grad, self.model.parameters()), cfg
        )
        self.scheduler = create_scheduler(self.optimizer, cfg)

        # Training state
        self.history = {
            'loss': [], 'top1_accuracy': [], 'top5_accuracy': [],
            'val_loss': [], 'val_top1_accuracy': [], 'val_top5_accuracy': []
        }
        self.best_val_top1 = 0.0

    def _setup_device(self):
        """Set up and return training device."""
        if self.cfg.training.device == 'auto':
            device = (
                torch.accelerator.current_accelerator().type
                if torch.accelerator.is_available() else 'cpu'
            )
        else:
            device = self.cfg.training.device
        print(f'Using {device} device')
        return device

    def _setup_directories(self):
        """Create output directories and return paths."""
        output_dir = 'outputs'
        checkpoints_dir = os.path.join(output_dir, 'checkpoints')
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            checkpoints_dir, self.cfg.training.checkpoint_path
        )

        # Auto-generate plot filename from checkpoint name
        base_name = os.path.splitext(
            self.cfg.training.checkpoint_path)[0]
        plot_filename = f'{base_name}.png'
        plot_path = os.path.join(plots_dir, plot_filename)

        return checkpoint_path, plot_path

    def train(self):
        """Execute the training loop."""
        print('[INFO] training network...')

        for epoch in range(self.cfg.training.epochs):
            # Train and validate
            train_loss, train_top1, train_top5 = train_one_epoch(
                self.model, self.train_loader, self.optimizer,
                self.criterion, self.device, self.num_classes
            )
            val_loss, val_top1, val_top5, _, _ = evaluate(
                self.model, self.val_loader, self.criterion,
                self.device, self.num_classes
            )

            # Update history
            self.history['loss'].append(train_loss)
            self.history['top1_accuracy'].append(train_top1)
            self.history['top5_accuracy'].append(train_top5)
            self.history['val_loss'].append(val_loss)
            self.history['val_top1_accuracy'].append(val_top1)
            self.history['val_top5_accuracy'].append(val_top5)

            # Step scheduler
            if self.scheduler:
                self.scheduler.step()

            # Log progress
            log_msg = (f'Epoch {epoch + 1}/{self.cfg.training.epochs} - '
                       f'loss: {train_loss:.4f} - top1: {train_top1:.2f}%')
            if train_top5 is not None:
                log_msg += f' - top5: {train_top5:.2f}%'
            log_msg += f' - val_loss: {val_loss:.4f}' \
                       f' - val_top1: {val_top1:.2f}%'
            if val_top5 is not None:
                log_msg += f' - val_top5: {val_top5:.2f}%'
            print(log_msg)

            # Save best model based on top-1 accuracy
            if val_top1 > self.best_val_top1:
                self.best_val_top1 = val_top1
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print('Saved best model')

    def evaluate(self):
        """Evaluate on test set and generate visualizations."""
        print('[INFO] evaluating network...')

        # Load best model
        self.model.load_state_dict(
            torch.load(self.checkpoint_path, map_location=self.device,
                       weights_only=True)
        )

        # Check crop_augmentation setting in config
        crop_aug = 'none'  # default
        if (hasattr(self.cfg.transforms, 'test') and
                hasattr(self.cfg.transforms.test, 'crop_augmentation')):
            crop_aug = self.cfg.transforms.test.crop_augmentation.lower()

        if crop_aug == 'ten_crop':
            # Use TenCrop evaluation (10 crops)
            print('[INFO] Evaluating with TenCrop augmentation (10 crops)...')
            _, test_top1, test_top5, predictions, labels = \
                evaluate_with_multicrop(
                    self.model, self.test_loader, self.criterion,
                    self.device, self.num_classes, num_crops=10
                )

            print(f'Test Top-1 Accuracy (TenCrop): {test_top1:.2f}%')
            if test_top5 is not None:
                print(f'Test Top-5 Accuracy (TenCrop): {test_top5:.2f}%')

            print('\nClassification Report (TenCrop):')
            print(classification_report(labels, predictions,
                                        target_names=self.class_names))

        elif crop_aug == 'five_crop':
            # Use FiveCrop evaluation (5 crops)
            print('[INFO] Evaluating with FiveCrop augmentation (5 crops)...')
            _, test_top1, test_top5, predictions, labels = \
                evaluate_with_multicrop(
                    self.model, self.test_loader, self.criterion,
                    self.device, self.num_classes, num_crops=5
                )

            print(f'Test Top-1 Accuracy (FiveCrop): {test_top1:.2f}%')
            if test_top5 is not None:
                print(f'Test Top-5 Accuracy (FiveCrop): {test_top5:.2f}%')

            print('\nClassification Report (FiveCrop):')
            print(classification_report(labels, predictions,
                                        target_names=self.class_names))

        else:
            # Use regular evaluation (no crop augmentation)
            print('[INFO] Evaluating with single center crop...')
            _, test_top1, test_top5, predictions, labels = \
                evaluate(
                    self.model, self.test_loader, self.criterion,
                    self.device, self.num_classes)

            print(f'Test Top-1 Accuracy: {test_top1:.2f}%')
            if test_top5 is not None:
                print(f'Test Top-5 Accuracy: {test_top5:.2f}%')

            print('\nClassification Report:')
            print(classification_report(labels, predictions,
                                        target_names=self.class_names))

        print(f'\nBest model saved to: {self.checkpoint_path}')

        # Plot training curves
        plot_training_history(self.history, self.plot_path)


def main(parsed_args):
    """Run main training pipeline."""
    cfg = Config(parsed_args.config)

    # Set random seed for reproducibility
    # pylint: disable=no-member
    set_seed(cfg.random_seed)
    # pylint: enable=no-member

    trainer = Trainer(cfg)
    trainer.train()
    trainer.evaluate()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Train image classification model')
    ap.add_argument('--config', type=str, required=True,
                    help='Path to config YAML file')
    args = ap.parse_args()
    # args = argparse.Namespace(config='./configs/cifar10_resnet50.yaml')
    # args = argparse.Namespace(config='./configs/dogs_vs_cats_alexnet.yaml')

    main(args)
