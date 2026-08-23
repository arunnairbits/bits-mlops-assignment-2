"""Evaluate a deployed Cats vs Dogs API with labeled image files."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import httpx

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
CLASS_NAMES = ("cat", "dog")


def class_name_from_path(path: Path) -> str | None:
    """Infer a binary class from any cat/dog directory component."""
    for component in reversed(path.parts[:-1]):
        normalized = component.lower().rstrip("s")
        if normalized in CLASS_NAMES:
            return normalized
    return None


def collect_samples(data_dir: Path, samples_per_class: int) -> list[tuple[Path, str]]:
    """Collect up to the requested number of images from each class directory."""
    samples: list[tuple[Path, str]] = []
    paths_by_class = {class_name: [] for class_name in CLASS_NAMES}
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            class_name = class_name_from_path(path)
            if class_name is not None:
                paths_by_class[class_name].append(path)
    for class_name in CLASS_NAMES:
        samples.extend((path, class_name) for path in paths_by_class[class_name][:samples_per_class])
    if not samples:
        raise FileNotFoundError(f"No cat/dog images found under {data_dir}.")
    return samples


def percentile(values: list[float], percentile_value: float) -> float:
    """Calculate a linearly interpolated percentile without extra dependencies."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def evaluate(api_url: str, data_dir: Path, samples_per_class: int) -> dict[str, object]:
    """Send labeled images to the API and return evaluation statistics."""
    samples = collect_samples(data_dir, samples_per_class)
    latencies: list[float] = []
    confusion = {actual: {predicted: 0 for predicted in CLASS_NAMES} for actual in CLASS_NAMES}
    correct = 0
    with httpx.Client(timeout=30.0) as client:
        for image_path, actual_class in samples:
            started = time.perf_counter()
            response = client.post(api_url, files={"file": (image_path.name, image_path.read_bytes(), "image/*")})
            latencies.append((time.perf_counter() - started) * 1000)
            response.raise_for_status()
            payload = response.json()
            predicted_class = str(payload.get("class", payload.get("label", ""))).lower()
            if predicted_class not in CLASS_NAMES:
                raise ValueError(f"Unexpected class in response for {image_path}: {predicted_class}")
            confusion[actual_class][predicted_class] += 1
            correct += predicted_class == actual_class
    return {
        "total_requests": len(samples),
        "average_latency_ms": statistics.fmean(latencies),
        "p95_latency_ms": percentile(latencies, 95),
        "overall_accuracy": correct / len(samples),
        "confusion_matrix": confusion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a deployed Cats vs Dogs prediction API.")
    parser.add_argument("--api-url", default="http://localhost:8000/predict")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--samples-per-class", type=int, default=20)
    args = parser.parse_args()
    summary = evaluate(args.api_url, args.data_dir, args.samples_per_class)
    print(f"Total requests: {summary['total_requests']}")
    print(f"Average latency (ms): {summary['average_latency_ms']:.2f}")
    print(f"P95 latency (ms): {summary['p95_latency_ms']:.2f}")
    print(f"Overall Accuracy: {summary['overall_accuracy']:.4f}")
    print("Confusion Matrix:")
    print(summary["confusion_matrix"])


if __name__ == "__main__":
    main()