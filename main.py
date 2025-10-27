"""
Image Classification Training Script.

Main entry point for training image classification models. Orchestrates the
entire training pipeline including data loading, model creation, training
loop, evaluation, and visualization.

Usage:
    python main.py --config configs/flowers17_vgg.yaml
    python main.py --config configs/flowers17_minivggnet.yaml

Command-line Arguments:
    --config: Path to YAML configuration file (required)

Outputs:
    - outputs/checkpoints/<model_name>.pth: Best model weights
    - outputs/plots/training_curves.png: Training/validation curves
"""


import argparse
import os

from data import get_data_loaders

from engine import evaluate, train_one_epoch

from models import create_model

from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

import torch
from torch.utils.data import DataLoader, Subset

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

        # Store class names for classification report
        self.class_names = self.train_loader.dataset.dataset.classes

        # Model components
        self.model = create_model(cfg).to(self.device)
        self.criterion = create_criterion(cfg)
        self.optimizer = create_optimizer(
            filter(lambda p: p.requires_grad, self.model.parameters()), cfg
        )
        self.scheduler = create_scheduler(self.optimizer, cfg)

        # Training state
        self.history = {
            'loss': [], 'accuracy': [],
            'val_loss': [], 'val_accuracy': []
        }
        self.best_val_acc = 0.0

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

    def tune_weight_decay(self):
        """
        Run k-fold cross-validation to find best weight_decay if configured.

        If optimizer config has cross_validation section, performs grid search
        and reinitializes model/optimizer with best weight_decay.
        Otherwise, uses weight_decay from config.
        """
        # Check if cross-validation is configured
        if not hasattr(self.cfg.optimizer, 'cross_validation'):
            print('[INFO] No cross-validation found in config file. '
                  f'Using weight_decay = {self.cfg.optimizer.weight_decay} '
                  ' from config')
            return

        print('[INFO] Starting cross-validation for weight_decay tuning...')
        # Run grid search with k-fold CV
        best_wd = self._run_cv_grid_search()

        print(f'[INFO] Best weight_decay from CV: {best_wd}')
        print('[INFO] Reinitializing model, optimizer, and scheduler with '
              'best weight_decay...')

        # Update config with best weight_decay
        self.cfg.optimizer.weight_decay = best_wd

        # Reinitialize model and optimizer
        self.model = create_model(self.cfg).to(self.device)
        self.optimizer = create_optimizer(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            self.cfg
        )
        # Recreate scheduler with new optimizer
        self.scheduler = create_scheduler(self.optimizer, self.cfg)

    def _train_single_fold(self, fold_indices, train_dataset,
                           weight_decay, fold):
        """
        Train and evaluate a single CV fold.

        Args:
            fold_indices: Tuple of (train_idx, val_idx) for this fold
            train_dataset: Full training dataset
            weight_decay: Weight decay value to use
            fold: Fold number for reproducibility

        Returns:
            float: Validation accuracy for this fold
        """
        train_idx, val_idx = fold_indices

        # Set seed for reproducibility within each fold
        set_seed(self.cfg.dataset.config.random_state + fold)

        # Create fold datasets
        fold_train = Subset(train_dataset, train_idx)
        fold_val = Subset(train_dataset, val_idx)

        fold_train_loader = DataLoader(
            fold_train, batch_size=self.cfg.dataset.batch_size,
            shuffle=True, num_workers=self.cfg.dataset.num_workers,
            pin_memory=True
        )
        fold_val_loader = DataLoader(
            fold_val, batch_size=self.cfg.dataset.batch_size,
            shuffle=False, num_workers=self.cfg.dataset.num_workers,
            pin_memory=True
        )

        # Create fresh model and optimizer for this fold
        fold_model = create_model(self.cfg).to(self.device)

        # Set weight_decay for this fold
        self.cfg.optimizer.weight_decay = weight_decay
        fold_optimizer = create_optimizer(
            filter(lambda p: p.requires_grad, fold_model.parameters()),
            self.cfg
        )

        # Train on this fold (silent - no epoch printing)
        cv_epochs = self.cfg.training.epochs
        for _ in range(cv_epochs):
            train_one_epoch(fold_model, fold_train_loader,
                            fold_optimizer, self.criterion, self.device)

        # Evaluate on validation fold
        _, val_acc, _, _ = evaluate(fold_model, fold_val_loader,
                                    self.criterion, self.device)

        # Cleanup to prevent memory issues
        del fold_model, fold_optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return val_acc

    def _run_cv_grid_search(self):
        """
        Run cross-validation grid search for weight_decay tuning.

        Returns:
            float: Best weight_decay value
        """
        weight_decay_grid = self.cfg.optimizer.cross_validation.weight_decay
        cv_folds = self.cfg.optimizer.cross_validation.folds

        train_dataset = self.train_loader.dataset
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        best_weight_decay = None
        best_mean_acc = 0.0

        print(f'Grid: {weight_decay_grid}')
        print(f'Folds: {cv_folds}. Epochs: {self.cfg.training.epochs}')

        # Grid search
        for wd in weight_decay_grid:
            fold_accuracies = []

            # K-fold cross-validation
            for fold, fold_indices in enumerate(
                    kf.split(range(len(train_dataset)))):

                val_acc = self._train_single_fold(
                    fold_indices, train_dataset, wd, fold
                )
                fold_accuracies.append(val_acc)

            # Compute mean accuracy across folds
            mean_acc = sum(fold_accuracies) / len(fold_accuracies)
            print(f'weight_decay={wd:.4f} => mean val acc: {mean_acc:.2f}%')

            if mean_acc > best_mean_acc:
                best_mean_acc = mean_acc
                best_weight_decay = wd

        print(f'Best mean validation accuracy: {best_mean_acc:.2f}%\n')
        return best_weight_decay

    def train(self):
        """Execute the training loop."""
        print('[INFO] training network...')

        for epoch in range(self.cfg.training.epochs):
            # Train and validate
            train_loss, train_acc = train_one_epoch(
                self.model, self.train_loader, self.optimizer,
                self.criterion, self.device
            )
            val_loss, val_acc, _, _ = evaluate(
                self.model, self.val_loader, self.criterion, self.device
            )

            # Update history
            self.history['loss'].append(train_loss)
            self.history['accuracy'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)

            # Step scheduler
            if self.scheduler:
                self.scheduler.step()

            # Log progress
            print(f'Epoch {epoch + 1}/{self.cfg.training.epochs} - '
                  f'loss: {train_loss:.4f} - acc: {train_acc:.2f}% - '
                  f'val_loss: {val_loss:.4f} - val_acc: {val_acc:.2f}%')

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
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

        # Evaluate on test set - get all 4 return values
        _, test_acc, predictions, labels = evaluate(
            self.model, self.test_loader, self.criterion, self.device
        )

        print(f'Test Accuracy: {test_acc:.2f}%')

        # Print classification report
        print(classification_report(labels, predictions,
                                    target_names=self.class_names))

        print(f'Best model saved to: {self.checkpoint_path}')

        # Plot training curves
        plot_training_history(self.history, self.plot_path)


def main(parsed_args):
    """Run main training pipeline."""
    cfg = Config(parsed_args.config)

    # Set random seed for reproducibility
    # pylint: disable=no-member
    set_seed(cfg.dataset.config.random_state, deterministic=True)
    # pylint: enable=no-member

    trainer = Trainer(cfg)
    trainer.tune_weight_decay()
    trainer.train()
    trainer.evaluate()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Train image classification model')
    ap.add_argument('--config', type=str, required=True,
                    help='Path to config YAML file')
    args = ap.parse_args()
    # args = argparse.Namespace(config='./configs/flowers17_minivggnet.yaml')

    main(args)
