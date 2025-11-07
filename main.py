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

    # Resume training
    python main.py --config configs/flowers17_vgg.yaml \
        --resume outputs/checkpoints/flowers17_vgg_last.pth

    # Resume with learning rate override
    python main.py --config configs/flowers17_vgg.yaml \
        --resume outputs/checkpoints/flowers17_vgg_last.pth \
        --override-lr 1e-5

Command-line Arguments:
    --config: Path to YAML configuration file (required)
    --resume: Path to checkpoint file to resume training from (optional)
    --override-lr: Override learning rate when resuming training (optional)

Outputs:
    - outputs/checkpoints/<dataset_name>_<backbone>.pth: Best model weights
    - outputs/checkpoints/<dataset_name>_<backbone>_last.pth: Last epoch checkpoint
    - outputs/checkpoints/<dataset_name>_<backbone>_epoch_N.pth: Periodic checkpoints
    - outputs/plots/<dataset_name>_<backbone>.png: Training/validation curves
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

    def __init__(self, cfg, resume_checkpoint=None, override_lr=None):
        """
        Initialize trainer with configuration.

        Args:
            cfg: Configuration object from YAML file
            resume_checkpoint: Path to checkpoint file to resume from (optional)
            override_lr: Learning rate to override when resuming (optional)
        """
        self.cfg = cfg
        self.device = self._setup_device()
        self.checkpoint_path, self.last_checkpoint_path, self.plot_path = \
            self._setup_directories()

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
        self.start_epoch = 0

        # Load checkpoint if resuming
        if resume_checkpoint:
            self._load_checkpoint(resume_checkpoint)

            # Override learning rate if specified
            if override_lr is not None:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = override_lr
                print(f'[INFO] Overriding learning rate to {override_lr}')

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

        # Auto-generate checkpoint names from dataset and model backbone
        base_filename = f'{self.cfg.dataset.name}_{self.cfg.model.backbone}'
        checkpoint_path = os.path.join(checkpoints_dir, f'{base_filename}_best.pth')
        last_checkpoint_path = os.path.join(checkpoints_dir, f'{base_filename}_last.pth')

        # Auto-generate plot filename
        plot_filename = f'{base_filename}.png'
        plot_path = os.path.join(plots_dir, plot_filename)

        return checkpoint_path, last_checkpoint_path, plot_path

    def _load_checkpoint(self, checkpoint_path):
        """
        Load checkpoint to resume training.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f'[INFO] Loading checkpoint from {checkpoint_path}...')
        checkpoint = torch.load(checkpoint_path, map_location=self.device,
                               weights_only=False)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print('[INFO] Model state loaded')

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('[INFO] Optimizer state loaded')

        # Load scheduler state if it exists
        if self.scheduler and 'scheduler_state_dict' in checkpoint and \
           checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print('[INFO] Scheduler state loaded')

        # Load training state
        self.start_epoch = checkpoint['epoch']
        self.best_val_top1 = checkpoint.get('best_val_top1', 0.0)
        self.history = checkpoint.get('history', self.history)

        print(f'[INFO] Resuming from epoch {self.start_epoch}')
        print(f'[INFO] Best validation top-1 accuracy so far: {self.best_val_top1:.2f}%')

    def _save_checkpoint(self, epoch, is_best=False, is_periodic=False):
        """
        Save checkpoint.

        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            is_periodic: Whether this is a periodic checkpoint (every 5 epochs)
        """
        checkpoint = {
            'epoch': epoch + 1,  # Save next epoch to resume from
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_top1': self.best_val_top1,
            'history': self.history
        }

        # Always save last checkpoint
        torch.save(checkpoint, self.last_checkpoint_path)

        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoint_path)
            print(f'[INFO] Saved best model to {self.checkpoint_path}')

        # Save periodic checkpoint (every 5 epochs)
        if is_periodic:
            periodic_path = self.checkpoint_path.replace('_best.pth', f'_epoch_{epoch+1}.pth')
            torch.save(checkpoint, periodic_path)
            print(f'[INFO] Saved periodic checkpoint to {periodic_path}')

    def train(self):
        """Execute the training loop."""
        print('[INFO] training network...')

        for epoch in range(self.start_epoch, self.cfg.training.epochs):
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

            current_lr = self.optimizer.param_groups[0]['lr']

            # Log progress
            log_msg = (f'Epoch {epoch + 1}/{self.cfg.training.epochs} - '
                       f'LR: {current_lr:.6f} - '
                       f'loss: {train_loss:.4f} - top1: {train_top1:.2f}%')
            if train_top5 is not None:
                log_msg += f' - top5: {train_top5:.2f}%'
            log_msg += f' - val_loss: {val_loss:.4f}' \
                       f' - val_top1: {val_top1:.2f}%'
            if val_top5 is not None:
                log_msg += f' - val_top5: {val_top5:.2f}%'
            print(log_msg)

            # Check if this is the best model
            is_best = val_top1 > self.best_val_top1
            if is_best:
                self.best_val_top1 = val_top1

            # Check if this is a periodic checkpoint (every 5 epochs)
            is_periodic = (epoch + 1) % 5 == 0

            # Save checkpoints
            self._save_checkpoint(epoch, is_best=is_best, is_periodic=is_periodic)

            # Update training plot after every epoch
            plot_training_history(self.history, self.plot_path)

    def evaluate(self):
        """Evaluate on test set and generate visualizations."""
        print('[INFO] evaluating network...')

        # Load best model
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device,
                           weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

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
        print(f'Training curves saved to: {self.plot_path}')


def main(parsed_args):
    """Run main training pipeline."""
    cfg = Config(parsed_args.config)

    # Set random seed for reproducibility
    # pylint: disable=no-member
    set_seed(cfg.random_seed)
    # pylint: enable=no-member

    trainer = Trainer(cfg, resume_checkpoint=parsed_args.resume,
                     override_lr=parsed_args.override_lr)
    trainer.train()
    trainer.evaluate()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Train image classification model')
    ap.add_argument('--config', type=str, required=True,
                    help='Path to config YAML file')
    ap.add_argument('--resume', type=str, default=None,
                    help='Path to checkpoint file to resume training from')
    ap.add_argument('--override-lr', type=float, default=None,
                    help='Override learning rate when resuming training')
    args = ap.parse_args()
    # args = argparse.Namespace(config='./configs/cifar10_resnet50.yaml')
    # args = argparse.Namespace(config='./configs/dogs_vs_cats_alexnet.yaml')
    # args = argparse.Namespace(config='./configs/tiny_imagenet200_deepergooglenet.yaml')

    main(args)
