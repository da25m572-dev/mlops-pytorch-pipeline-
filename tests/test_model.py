"""Unit tests for src/model.py."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import SimpleCNN, get_model  # noqa: E402


@pytest.mark.parametrize("architecture", ["resnet18", "simple_cnn"])
def test_get_model_forward_shape(architecture: str) -> None:
    num_classes = 10
    model = get_model(architecture=architecture, num_classes=num_classes)
    model.eval()

    batch = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        logits = model(batch)

    assert logits.shape == (4, num_classes)


def test_get_model_unknown_architecture_raises() -> None:
    with pytest.raises(ValueError):
        get_model(architecture="not-a-real-model", num_classes=10)


def test_simple_cnn_has_trainable_parameters() -> None:
    model = SimpleCNN(num_classes=10)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_params) > 0


def test_get_model_num_classes_respected() -> None:
    model = get_model(architecture="simple_cnn", num_classes=5)
    batch = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (2, 5)

