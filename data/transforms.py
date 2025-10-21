from torchvision import transforms

def get_transforms(transform_cfg):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Training transforms with augmentation
    train_tf = []

    # Use RandomResizedCrop if configured, otherwise fall back to Resize + RandomCrop
    if getattr(transform_cfg, 'random_resized_crop', False):
        train_tf.append(transforms.RandomResizedCrop(transform_cfg.crop))
    else:
        train_tf.extend([
            transforms.Resize(transform_cfg.resize),
            transforms.RandomCrop(transform_cfg.crop)
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
