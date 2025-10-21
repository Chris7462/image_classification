import torch.nn as nn
from torchvision import models
from models.custom.minivggnet import MiniVGGNet

def create_model(cfg):
    backbone = cfg.model.backbone.lower()
    num_classes = cfg.model.num_classes

    if backbone == "minivggnet":
        # Custom MiniVGGNet doesn't use pretrained weights
        model = MiniVGGNet(num_classes=num_classes)

    elif backbone == "vgg16":
        pretrained = cfg.model.pretrained
        freeze_backbone = cfg.model.freeze_backbone
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        if freeze_backbone:
            for p in model.features.parameters():
                p.requires_grad = False
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif backbone == "resnet50":
        pretrained = cfg.model.pretrained
        freeze_backbone = cfg.model.freeze_backbone
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        if freeze_backbone:
            for p in list(model.children())[:-1]:
                for param in p.parameters():
                    param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return model
