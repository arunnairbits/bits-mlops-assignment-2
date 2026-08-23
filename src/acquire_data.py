"""Download the Kaggle Cats vs Dogs dataset into the DVC-tracked raw data folder."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from kaggle import KaggleApi

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DEFAULT_DATASET = "bhavikjikadara/dog-and-cat-classification-dataset"


def ensure_data_directories() -> None:
    """Create the raw and processed data directories if they do not exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset(dataset_name: str = DEFAULT_DATASET, output_dir: Path = RAW_DIR) -> Path:
    """Download the Kaggle dataset and unzip it into the raw directory."""
    ensure_data_directories()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_name, path=str(output_dir), unzip=True)

    print(f"Dataset downloaded to {output_dir}")
    return output_dir


def main() -> None:
    """Run the Kaggle dataset download using the environment override when present."""
    dataset_name = os.getenv("KAGGLE_DATASET", DEFAULT_DATASET)
    download_dataset(dataset_name=dataset_name, output_dir=RAW_DIR)


if __name__ == "__main__":
    main()
