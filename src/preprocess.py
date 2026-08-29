"""Image preprocessing and dataset splitting utilities for Cats vs Dogs."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

IMAGE_SIZE = (224, 224)
DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_BATCH_SIZE = 32


def preprocess_image(image: Image.Image) -> Image.Image:
    """Convert an image to RGB and resize it to the model input dimensions."""
    return image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)


class PreprocessedImageFolder(datasets.ImageFolder):
    """ImageFolder variant that guarantees RGB, 224x224 samples."""

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = preprocess_image(self.loader(path))
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Build training augmentation and deterministic evaluation transforms."""
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.ToTensor(), normalize]
    )
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])
    return train_transform, eval_transform


def split_indices(dataset_size: int, seed: int = 42) -> tuple[list[int], list[int], list[int]]:
    """Return deterministic 80/10/10 train, validation, and test indices."""
    if dataset_size < 3:
        raise ValueError("The image dataset must contain at least three images.")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(dataset_size, generator=generator).tolist()
    train_end = max(1, min(int(dataset_size * 0.8), dataset_size - 2))
    validation_end = max(train_end + 1, min(train_end + int(dataset_size * 0.1), dataset_size - 1))
    return permutation[:train_end], permutation[train_end:validation_end], permutation[validation_end:]


def build_datasets(data_dir: Path = DEFAULT_DATA_DIR, seed: int = 42) -> tuple[Dataset, Dataset, Dataset, list[str]]:
    """Load an ImageFolder dataset and split it into train, validation, and test sets."""
    train_transform, eval_transform = build_transforms()
    train_source = PreprocessedImageFolder(str(data_dir), transform=train_transform)
    eval_source = PreprocessedImageFolder(str(data_dir), transform=eval_transform)
    train_indices, validation_indices, test_indices = split_indices(len(train_source), seed)
    return Subset(train_source, train_indices), Subset(eval_source, validation_indices), Subset(eval_source, test_indices), train_source.classes


def build_dataloaders(
    data_dir: Path = DEFAULT_DATA_DIR, batch_size: int = DEFAULT_BATCH_SIZE, seed: int = 42
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Create shuffled training and deterministic validation/test loaders."""
    train_dataset, validation_dataset, test_dataset, classes = build_datasets(data_dir, seed)
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(validation_dataset, batch_size=batch_size),
        DataLoader(test_dataset, batch_size=batch_size),
        classes,
    )
    