"""Inference helpers for the saved Cats vs Dogs CNN."""

from __future__ import annotations

import io
import os
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from src.preprocess import build_transforms, preprocess_image
from src.train import BaselineCNN

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/cats_vs_dogs_cnn.pt"))


def load_model(model_path: Path = MODEL_PATH) -> tuple[BaselineCNN, list[str]]:
    """Load the serialized CNN and its class names from disk."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run `uv run python src/train.py` first."
        )
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    model = BaselineCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes


def predict_image(
    image_bytes: bytes, model: BaselineCNN, classes: list[str]
) -> dict[str, str | int | float]:
    """Predict a class from raw image bytes using a loaded CNN."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image_tensor = build_transforms()[1](preprocess_image(image)).unsqueeze(0)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("The uploaded file is not a valid image.") from error

    with torch.inference_mode():
        probabilities = torch.softmax(model(image_tensor), dim=1)[0]
    class_index = int(probabilities.argmax().item())
    return {
        "prediction": class_index,
        "label": classes[class_index],
        "confidence": round(float(probabilities[class_index].item()), 4),
    }