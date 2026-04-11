from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


def create_mobilenetv2_classifier(
    num_classes: int = 1,
    pretrained: bool = True,
) -> nn.Module:
    """
    Create MobileNetV2 with binary classification head.

    Args:
        num_classes: Number of output units (1 for binary with BCEWithLogitsLoss)
        pretrained: Whether to use ImageNet pretrained weights

    Returns:
        MobileNetV2 model with custom classification head
    """
    model = mobilenet_v2(pretrained=pretrained, progress=True)

    # Replace classifier
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, num_classes),
    )

    return model


def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    """Freeze or unfreeze backbone (all layers except classifier)."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = not freeze
