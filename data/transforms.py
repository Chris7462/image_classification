"""
Image Transformation Module.

Defines transformation pipelines for training and validation/test datasets.
Training transforms include data augmentation techniques, while validation/test
transforms apply only deterministic preprocessing.

Supported Augmentations:
    - Random resized crop with configurable scale or resize + random crop
    - Random affine transformations (rotation, translation, shear, scale)
    - Random horizontal flip
    - Color jitter (brightness, contrast, saturation, hue)
    - Random grayscale conversion

All transforms include ImageNet normalization (mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]) for transfer learning compatibility.

Functions:
    get_transforms: Build train and validation transform pipelines from config

Example:
    >>> from utils import Config
    >>> from data import get_transforms
    >>> cfg = Config('configs/flowers17_vgg.yaml')
    >>> train_tf, val_tf = get_transforms(cfg.dataset.transforms)
"""

from torchvision import transforms


def _get_transforms(transform_cfg):
    """
    Build training and validation transform pipelines from configuration.

    Creates two separate transform pipelines:
    - Training: Includes data augmentation for regularization
    - Validation/Test: Deterministic preprocessing only

    Args:
        transform_cfg: Configuration object containing transform parameters:
            - resize (int): Target size for resizing
            - crop (int): Size for cropping (used when random_resized_crop not provided)
            - normalize (dict, optional): Normalization parameters
                (defaults to ImageNet values)
                - mean (list): RGB mean values
                - std (list): RGB standard deviation values
            - random_resized_crop (dict, optional): RandomResizedCrop parameters
                - size (int): Output size
                - scale (list, optional): Scale range [min, max] (default: [0.08, 1.0])
                - ratio (list, optional): Aspect ratio range (default: [3/4, 4/3])
            - random_affine (dict, optional): Affine transformation parameters
                - degrees (float): Rotation range
                - translate (list): Translation range [x, y]
                - shear (float): Shear range
                - scale (list): Scaling range [min, max]
            - random_horizontal_flip (bool, optional): Enable horizontal flip
            - color_jitter (dict, optional): Color jitter parameters
                - brightness (float): Brightness adjustment range
                - contrast (float): Contrast adjustment range
                - saturation (float): Saturation adjustment range
                - hue (float): Hue adjustment range
            - random_grayscale (dict, optional): Grayscale conversion
                parameters
                - p (float): Probability of conversion

    Returns:
        tuple: (train_transforms, val_transforms) - Composed transform pipeline

    Example:
        >>> from types import SimpleNamespace
        >>> cfg = SimpleNamespace(
        ...     resize=256, crop=224,
        ...     random_resized_crop=SimpleNamespace(
        ...         size=224,
        ...         scale=[0.7, 1.0]
        ...     ),
        ...     random_horizontal_flip=True,
        ...     normalize=SimpleNamespace(
        ...         mean=[0.485, 0.456, 0.406],
        ...         std=[0.229, 0.224, 0.225]
        ...     )
        ... )
        >>> train_tf, val_tf = get_transforms(cfg)
        >>> # Apply to image
        >>> train_img = train_tf(image)
    """
    # Get normalization parameters (default to ImageNet values)
    if hasattr(transform_cfg, 'normalize'):
        mean = transform_cfg.normalize.mean
        std = transform_cfg.normalize.std
    else:
        # Default to ImageNet values for backward compatibility
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    # Training transforms with augmentation
    train_tf = []

    # Use RandomResizedCrop if configured
    random_resized_crop = getattr(transform_cfg, 'random_resized_crop', None)

    if random_resized_crop:
        # Extract parameters from config
        size = random_resized_crop.size
        scale = getattr(random_resized_crop, 'scale', [0.08, 1.0])
        ratio = getattr(random_resized_crop, 'ratio', [3./4., 4./3.])
        train_tf.append(transforms.RandomResizedCrop(
            size=size,
            scale=tuple(scale),
            ratio=tuple(ratio)
        ))
    else:
        # Fall back to Resize + CenterCrop
        train_tf.extend([
            transforms.Resize(transform_cfg.resize),
            transforms.CenterCrop(transform_cfg.crop)
        ])

    # Add RandomAffine if configured
    random_affine = getattr(transform_cfg, 'random_affine', None)
    if random_affine:
        train_tf.append(transforms.RandomAffine(
            degrees=random_affine.degrees,
            translate=tuple(random_affine.translate),
            shear=random_affine.shear,
            scale=tuple(random_affine.scale)
        ))

    # Add horizontal flip if configured
    if getattr(transform_cfg, 'random_horizontal_flip', False):
        train_tf.append(transforms.RandomHorizontalFlip())

    # Add ColorJitter if configured
    color_jitter = getattr(transform_cfg, 'color_jitter', None)
    if color_jitter:
        train_tf.append(transforms.ColorJitter(
            brightness=getattr(color_jitter, 'brightness', 0),
            contrast=getattr(color_jitter, 'contrast', 0),
            saturation=getattr(color_jitter, 'saturation', 0),
            hue=getattr(color_jitter, 'hue', 0)
        ))

    # Add RandomGrayscale if configured
    random_grayscale = getattr(transform_cfg, 'random_grayscale', None)
    if random_grayscale:
        train_tf.append(transforms.RandomGrayscale(p=random_grayscale.p))

    train_tf.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_tf = transforms.Compose(train_tf)

    # Validation transforms (no augmentation, deterministic)
    val_tf = [
        transforms.Resize(transform_cfg.resize),
        transforms.CenterCrop(transform_cfg.crop),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
    val_tf = transforms.Compose(val_tf)

    return train_tf, val_tf
