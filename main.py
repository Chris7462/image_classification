import os
import argparse
import torch

from data import get_data_loaders
from models import create_model
from engine import train_one_epoch, evaluate
from utils import Config
from utils import create_criterion, create_optimizer, create_scheduler
from utils import plot_training_history


def main(args):
    # Load config from command line argument
    cfg = Config(args.config)

    # Device selection: support 'auto' for automatic detection or explicit device
    if cfg.training.device == 'auto':
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    else:
        device = cfg.training.device
    print(f"Using {device} device")

    # Create output directories
    output_dir = "outputs"
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    best_model_path = os.path.join(checkpoints_dir, cfg.training.checkpoint_path)

    # Data
    train_loader, val_loader, test_loader = get_data_loaders(cfg)

    # Model
    model = create_model(cfg).to(device)

    # Loss, optimizer, and scheduler from config
    criterion = create_criterion(cfg)
    optimizer = create_optimizer(filter(lambda p: p.requires_grad, model.parameters()), cfg)
    scheduler = create_scheduler(optimizer, cfg)

    # Training history
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    # Training loop
    print("[INFO] training network...")
    best_val_acc = 0
    for epoch in range(cfg.training.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Update history
        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch+1}/{cfg.training.epochs} - "
              f"loss: {train_loss:.4f} - acc: {train_acc:.2f}% - "
              f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print("✅ Saved best model")

    # Test evaluation
    print("[INFO] evaluating network...")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"📊 Test Accuracy: {test_acc:.2f}%")
    print(f"💾 Best model saved to: {best_model_path}")

    # Plot training curves
    plot_path = os.path.join(plots_dir, "training_curves.png")
    plot_training_history(history, plot_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Train image classification model')
    ap.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = ap.parse_args()

    main(args)
