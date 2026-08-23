"""Tests for the CNN model interface."""

import torch

from src.train import BaselineCNN


def test_baseline_cnn_forward_returns_two_class_logits() -> None:
    batch_size = 4
    model = BaselineCNN(num_classes=2)

    output = model(torch.zeros(batch_size, 3, 224, 224))

    assert output.shape == (batch_size, 2)
