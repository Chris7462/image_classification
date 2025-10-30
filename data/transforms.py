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

All transforms include normalization with configurable mean/std values
(defaults to ImageNet normalization for transfer learning compatibility).

Functions:
    get_transforms: Build train, validation, and test transform pipelines from config

Example:
    >>> from utils import Config
    >>> from data import get_transforms
    >>> cfg = Config('configs/flowers17_vgg.yaml')
    >>> train_tf, val_tf, test_tf = get_transforms(cfg.transforms)
"""

from torchvision import transforms


def _get_transforms(transform_cfg):
    """
    Build training, validation, and test transform pipelines from configuration.

    Creates three separate transform pipelines based on the config structure:
    - Training: Includes data augmentation for regularization
    - Validation: Deterministic preprocessing (uses common transforms)
    - Test: Deterministic preprocessing (uses common transforms)

    The transform application follows this logic:
    1. Apply split-specific transforms (train/val/test)
    2. If split doesn't define resize/crop, use common resize/crop
    3. Apply common normalize at the end

    Args:
        transform_cfg: Configuration object containing:
            - common: Common settings for all splits
                - resize (int): Target size for resizing
                - crop (int): Size for center cropping
                - normalize (dict, optional): Normalization parameters
                    - mean (list): RGB mean values
                    - std (list): RGB standard deviation values
            - train: Training-specific transforms
                - random_resized_crop (dict, optional): RandomResizedCrop parameters
                    - size (int): Output size
                    - scale (list, optional): Scale range [min, max]
                    - ratio (list, optional): Aspect ratio range
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
                    - p (float): Probability of conversion
            - val: Validation-specific transforms (null to use only common)
            - test: Test-specific transforms (null to use only common)

    Returns:
        tuple: (train_transforms, val_transforms, test_transforms) - Composed pipelines

    Example:
        >>> from types import SimpleNamespace
        >>> cfg = SimpleNamespace(
        ...     common=SimpleNamespace(
        ...         resize=256,
        ...         crop=224,
        ...         normalize=SimpleNamespace(
        ...             mean=[0.485, 0.456, 0.406],
        ...             std=[0.229, 0.224, 0.225]
        ...         )
        ...     ),
        ...     train=SimpleNamespace(
        ...         random_resized_crop=SimpleNamespace(size=224, scale=[0.7, 1.0]),
        ...         random_horizontal_flip=True
        ...     ),
        ...     val=None,
        ...     test=None
        ... )
        >>> train_tf, val_tf, test_tf = get_transforms(cfg)
    """
    # Get normalization parameters from common config
    common_cfg = transform_cfg.common
    if hasattr(common_cfg, 'normalize'):
        mean = common_cfg.normalize.mean
        std = common_cfg.normalize.std
    else:
        # Default to ImageNet values
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    # Build training transforms
    train_cfg = getattr(transform_cfg, 'train', None)
    train_tf = []
    if train_cfg and hasattr(train_cfg, 'random_resized_crop'):
        # Use random_resized_crop (overrides common resize + crop)
        rrc = train_cfg.random_resized_crop
        train_tf.append(transforms.RandomResizedCrop(
            size=rrc.size,
            scale=tuple(getattr(rrc, 'scale', [0.08, 1.0])),
            ratio=tuple(getattr(rrc, 'ratio', [3./4., 4./3.]))
        ))
    else:
        # Use common resize and crop
        train_tf.extend([
            transforms.Resize(common_cfg.resize),
            transforms.CenterCrop(common_cfg.crop)
        ])

    # Add training augmentations
    if train_cfg:
        if hasattr(train_cfg, 'random_affine'):
            ra = train_cfg.random_affine
            train_tf.append(transforms.RandomAffine(
                degrees=getattr(ra, 'degrees', 0),
                translate=tuple(getattr(ra, 'translate', [0.0, 0.0])),
                shear=getattr(ra, 'shear', 0),
                scale=tuple(getattr(ra, 'scale', [1.0, 1.0]))
            ))

        if getattr(train_cfg, 'random_horizontal_flip', False):
            train_tf.append(transforms.RandomHorizontalFlip())

        if hasattr(train_cfg, 'color_jitter'):
            cj = train_cfg.color_jitter
            train_tf.append(transforms.ColorJitter(
                brightness=getattr(cj, 'brightness', 0),
                contrast=getattr(cj, 'contrast', 0),
                saturation=getattr(cj, 'saturation', 0),
                hue=getattr(cj, 'hue', 0)
            ))

        if hasattr(train_cfg, 'random_grayscale'):
            rg = train_cfg.random_grayscale
            train_tf.append(transforms.RandomGrayscale(p=rg.p))

    train_tf.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_tf = transforms.Compose(train_tf)

    # Build validation transforms (uses common resize + crop)
    # val_cfg = getattr(transform_cfg, 'val', None)
    val_tf = [
        transforms.Resize(common_cfg.resize),
        transforms.CenterCrop(common_cfg.crop),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
    val_tf = transforms.Compose(val_tf)

    # Build test transforms (uses common resize + crop)
    # test_cfg = getattr(transform_cfg, 'test', None)
    test_tf = [
        transforms.Resize(common_cfg.resize),
        transforms.CenterCrop(common_cfg.crop),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
    test_tf = transforms.Compose(test_tf)

    return train_tf, val_tf, test_tf
