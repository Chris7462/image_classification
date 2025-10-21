import torch
from data.datasets import get_data_loaders
from models.model_factory import create_model
from utils.optimizer_factory import create_criterion, create_optimizer, create_scheduler
from engine.trainer import train_one_epoch, evaluate
from utils.config import Config


def main():
    cfg = Config("configs/flowers17_vgg.yaml")
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, test_loader = get_data_loaders(cfg)

    # Model
    model = create_model(cfg).to(device)

    # Loss, optimizer, and scheduler from config
    criterion = create_criterion(cfg)
    optimizer = create_optimizer(filter(lambda p: p.requires_grad, model.parameters()), cfg)
    scheduler = create_scheduler(optimizer, cfg)

    # Training loop
    best_val_acc = 0
    for epoch in range(cfg.training.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if scheduler:
            scheduler.step()

        print(f"Epoch [{epoch+1}/{cfg.training.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), cfg.training.checkpoint_path)
            print("✅ Saved best model")

    # Test evaluation
    model.load_state_dict(torch.load(cfg.training.checkpoint_path))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"📊 Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
