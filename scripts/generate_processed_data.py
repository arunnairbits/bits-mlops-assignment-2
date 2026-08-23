"""Generate DVC-tracked 224x224 RGB images from the raw dataset."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.preprocess import preprocess_image

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def generate_processed_data(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> int:
    """Resize every supported raw image while preserving its relative path."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_count = 0
    for source_path in raw_dir.rglob("*"):
        if not source_path.is_file() or source_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        destination_path = processed_dir / source_path.relative_to(raw_dir)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(source_path) as image:
                preprocess_image(image).save(destination_path)
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError(f"Unable to process image: {source_path}") from error
        processed_count += 1
    return processed_count


if __name__ == "__main__":
    count = generate_processed_data()
    print(f"Generated {count} processed images in {PROCESSED_DIR}")
