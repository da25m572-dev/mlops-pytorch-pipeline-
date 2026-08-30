"""Model definitions for the mlops-pytorch-pipeline image classifier.

Supports a lightweight custom CNN and a CIFAR-adapted ResNet-18, selected via
the ``model.architecture`` key in ``configs/training_config.yaml``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


class SimpleCNN(nn.Module):
    """A small CNN classifier for 32x32 RGB images (CIFAR-10 style)."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def _build_resnet18(num_classes: int) -> nn.Module:
    """ResNet-18 adapted for small (32x32) images such as CIFAR-10.

    The stock torchvision stem (7x7 stride-2 conv + maxpool) is tuned for
    224x224 ImageNet inputs and downsamples 32x32 inputs too aggressively.
    We swap in a 3x3 stride-1 conv and drop the initial maxpool, the standard
    adaptation used for CIFAR-scale ResNets.
    """
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    """Factory returning a classifier by architecture name.

    Args:
        architecture: One of "resnet18" or "simple_cnn".
        num_classes: Number of output classes.

    Returns:
        An initialized ``nn.Module``.

    Raises:
        ValueError: If ``architecture`` is not recognized.
    """
    architecture = architecture.lower()
    if architecture == "resnet18":
        return _build_resnet18(num_classes=num_classes)
    if architecture in ("simple_cnn", "cnn"):
        return SimpleCNN(num_classes=num_classes)
    raise ValueError(f"Unknown architecture: {architecture!r}")
