"""
Hyperparameter Optimization Script.

Performs k-fold cross-validation to find optimal weight_decay value for
training. Results can be manually added to config file before final training.

Usage:
    python scripts/optimize_hyperparameters.py \
        --config configs/caltech-101_vgg16_logistic.yaml

Output:
    Prints best weight_decay value based on cross-validation results.
"""

import argparse

from data import get_data_loaders

from engine import evaluate, train_one_epoch

from models import create_model

from sklearn.model_selection import KFold

import torch
from torch.utils.data import DataLoader, Subset

from utils import Config, create_criterion, create_optimizer, set_seed


# pylint: disable=too-many-arguments,too-many-locals
def train_single_fold(cfg, fold_indices, train_dataset, weight_decay, fold,
                      device, criterion):
    """
    Train and evaluate a single CV fold.

    Args:
        cfg: Configuration object
        fold_indices: Tuple of (train_idx, val_idx) for this fold
        train_dataset: Full training dataset
        weight_decay: Weight decay value to use
        fold: Fold number for reproducibility
        device: Device to run training on
        criterion: Loss function

    Returns:
        float: Validation accuracy for this fold
    """
    train_idx, val_idx = fold_indices

    # Set seed for reproducibility within each fold
    set_seed(cfg.dataset.config.random_state + fold)

    # Create fold datasets
    fold_train = Subset(train_dataset, train_idx)
    fold_val = Subset(train_dataset, val_idx)

    fold_train_loader = DataLoader(
        fold_train, batch_size=cfg.dataset.batch_size,
        shuffle=True, num_workers=cfg.dataset.num_workers,
        pin_memory=True
    )
    fold_val_loader = DataLoader(
        fold_val, batch_size=cfg.dataset.batch_size,
        shuffle=False, num_workers=cfg.dataset.num_workers,
        pin_memory=True
    )

    # Create fresh model and optimizer for this fold
    fold_model = create_model(cfg).to(device)

    # Set weight_decay for this fold
    cfg.optimizer.weight_decay = weight_decay
    fold_optimizer = create_optimizer(
        filter(lambda p: p.requires_grad, fold_model.parameters()), cfg
    )

    # Train on this fold (silent - no epoch printing)
    for _ in range(cfg.training.epochs):
        train_one_epoch(fold_model, fold_train_loader,
                        fold_optimizer, criterion, device)

    # Evaluate on validation fold
    _, val_acc, _, _ = evaluate(fold_model, fold_val_loader,
                                criterion, device)

    # Cleanup to prevent memory issues
    del fold_model, fold_optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return val_acc


# pylint: disable=too-many-locals
def run_cv_grid_search(cfg, train_loader, device, criterion):
    """
    Run cross-validation grid search for weight_decay tuning.

    Args:
        cfg: Configuration object
        train_loader: Training data loader
        device: Device to run training on
        criterion: Loss function

    Returns:
        float: Best weight_decay value
    """
    weight_decay_grid = cfg.optimizer.cross_validation.weight_decay
    cv_folds = cfg.optimizer.cross_validation.folds

    train_dataset = train_loader.dataset
    kf = KFold(n_splits=cv_folds, shuffle=True,
               random_state=cfg.dataset.config.random_state)

    best_weight_decay = None
    best_mean_acc = 0.0

    print(f'Grid: {weight_decay_grid}')
    print(f'Folds: {cv_folds}. Epochs: {cfg.training.epochs}')

    # Grid search
    for wd in weight_decay_grid:
        fold_accuracies = []

        # K-fold cross-validation
        for fold, fold_indices in enumerate(
                kf.split(range(len(train_dataset)))):

            val_acc = train_single_fold(
                cfg, fold_indices, train_dataset, wd, fold, device, criterion
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


def main():
    """Run hyperparameter optimization."""
    parser = argparse.ArgumentParser(
        description='Optimize weight_decay using cross-validation'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    args = parser.parse_args()

    print(f'Loading configuration from: {args.config}')
    cfg = Config(args.config)

    # Check if cross-validation is configured
    # pylint: disable=no-member
    if not hasattr(cfg.optimizer, 'cross_validation'):
        print('[ERROR] No cross_validation section found in config file.')
        print('Please add cross_validation section with weight_decay grid '
              'and folds.')
        return

    # Set random seed for reproducibility
    set_seed(cfg.dataset.config.random_state)

    # Setup device
    if cfg.training.device == 'auto':
        device = (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available() else 'cpu'
        )
    else:
        device = cfg.training.device
    # pylint: enable=no-member
    print(f'Using {device} device\n')

    # Load data
    print('[INFO] Loading data...')
    train_loader, _, _ = get_data_loaders(cfg)

    # Create criterion
    criterion = create_criterion(cfg)

    # Run cross-validation
    print('[INFO] Starting cross-validation for weight_decay tuning...')
    best_wd = run_cv_grid_search(cfg, train_loader, device, criterion)

    # Print results
    print('='*60)
    print('Optimization Results')
    print('='*60)
    print(f'Best weight_decay: {best_wd}')
    print('='*60)
    print('\nUpdate your config file:')
    print('optimizer:')
    print(f'  weight_decay: {best_wd}')
    print('='*60)


if __name__ == '__main__':
    main()
