"""Tests for the CNN model and image inference interface."""

import io

import torch
from PIL import Image

from src.predict import predict_image
from src.train import BaselineCNN


def test_baseline_cnn_forward_returns_two_class_logits() -> None:
    batch_size = 4
    model = BaselineCNN(num_classes=2)

    output = model(torch.zeros(batch_size, 3, 224, 224))

    assert output.shape == (batch_size, 2)


def test_predict_image_inference_utility_accepts_raw_image_bytes() -> None:
    image_buffer = io.BytesIO()
    Image.new("RGB", (32, 48), color="red").save(image_buffer, format="PNG")

    result = predict_image(image_buffer.getvalue(), BaselineCNN(), ["cat", "dog"])

    assert result["prediction"] in {0, 1}
    assert result["label"] in {"cat", "dog"}
    assert 0.0 <= result["confidence"] <= 1.0
