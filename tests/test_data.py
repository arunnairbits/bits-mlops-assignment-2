"""Tests for image preprocessing and dataset splitting."""

from PIL import Image

from src.preprocess import preprocess_image, split_indices


def test_preprocess_image_returns_224_rgb_image() -> None:
    image = Image.new("L", (80, 120), color=128)

    processed = preprocess_image(image)

    assert processed.mode == "RGB"
    assert processed.size == (224, 224)


def test_split_indices_is_deterministic_and_80_10_10() -> None:
    first_split = split_indices(100, seed=42)
    second_split = split_indices(100, seed=42)

    assert first_split == second_split
    assert tuple(len(indices) for indices in first_split) == (80, 10, 10)
    assert len(set().union(*map(set, first_split))) == 100
