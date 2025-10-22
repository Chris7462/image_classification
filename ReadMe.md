# Image Classification

A modular PyTorch-based framework for training image classification models with support for custom architectures and transfer learning.

## Features

- **Multiple Architectures**: MiniVGGNet, VGG16, ResNet50
- **Transfer Learning**: Pretrained weights with optional backbone freezing
- **Flexible Data Pipeline**: Stratified splits, extensive augmentation options
- **YAML Configuration**: Easy experiment management
- **Dataset Support**: Flowers17, CIFAR10, extensible to custom datasets

## Project Structure

```
image_classification/
├── configs/           # YAML configuration files
├── data/              # Data loading and preprocessing
├── engine/            # Training and evaluation loops
├── models/            # Model architectures and factory
│   └── custom/        # Custom model implementations
├── utils/             # Utilities (config, optimizer, visualization)
└── scripts/           # Helper scripts
```

## Module Overview

### `data`
Handles dataset loading with stratified train/val/test splits and transforms.
- `data_loaders.py`: DataLoader creation with configurable splits
- `transforms.py`: Training augmentation and validation preprocessing

### `models`
Model creation with support for custom and pretrained architectures.
- `model_factory.py`: Factory for instantiating models from config
- `custom/minivggnet.py`: Lightweight CNN for small datasets

### `engine`
Core training and evaluation functionality.
- `train_eval.py`: Single-epoch training and evaluation loops

### `utils`
Supporting utilities for the training pipeline.
- `config_parser.py`: YAML config with dot notation access
- `optimizer_factory.py`: Loss, optimizer, and scheduler creation
- `visualization.py`: Training curve plotting
- `seed.py`: Reproducibility utilities

### `scripts`
Helper scripts for dataset analysis.
- `calculate_dataset_stats.py`: Compute mean/std for normalization

## Installation

```bash
pip install torch torchvision pyyaml scikit-learn matplotlib numpy
```

## Usage

### Basic Training

```bash
python main.py --config configs/flowers17_vgg.yaml
```

### Training Examples

**Transfer Learning with VGG16:**
```bash
python main.py --config configs/flowers17_vgg.yaml
```

**Training from Scratch with MiniVGGNet:**
```bash
python main.py --config configs/flowers17_minivggnet.yaml
```

**CIFAR10 with ResNet50:**
```bash
python main.py --config configs/cifar10_resnet50.yaml
```

### Calculate Dataset Statistics

```bash
python scripts/calculate_dataset_stats.py --config configs/flowers17_vgg.yaml
```

## Configuration

Example configuration structure:

```yaml
dataset:
  name: flowers17
  batch_size: 32
  config:
    data_root: /path/to/dataset
    split_ratios:
      train: 0.7
      val: 0.15
      test: 0.15

model:
  backbone: vgg16
  num_classes: 17
  pretrained: true
  freeze_backbone: true

optimizer:
  type: adam
  lr: 0.0001

training:
  epochs: 10
  device: auto
```

## Output

Training produces:
- `outputs/checkpoints/<model_name>.pth`: Best model weights
- `outputs/plots/<model_name>.png`: Training curves

## Extending the Framework

### Adding a New Dataset

1. Implement loader function in `data/data_loaders.py`
2. Add case in `get_data_loaders()`
3. Create config file in `configs/`

### Adding a New Model

1. Implement architecture in `models/custom/`
2. Add case in `models/model_factory.py`
3. Update config with new backbone name

## License

MIT License
