"""Train and evaluate a baseline CNN for Cats vs Dogs classification."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import torch
from PIL import ImageFile
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.preprocess import DEFAULT_DATA_DIR, build_dataloaders

ImageFile.LOAD_TRUNCATED_IMAGES = True

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/cats_vs_dogs_cnn.pt"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")
MLFLOW_EXPERIMENT_NAME = "cats-vs-dogs-image-classification"


class BaselineCNN(nn.Module):
    """Compact CNN suitable as a reproducible assignment baseline."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images).flatten(1))


@dataclass
class TrainingResult:
    model_path: str
    classes: list[str]
    test_loss: float
    test_accuracy: float


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: Adam | None = None) -> tuple[float, float]:
    """Run one training or evaluation epoch and return average loss and accuracy."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    if total == 0:
        raise ValueError("A dataset split is empty; provide at least three images.")
    return total_loss / total, correct / total


def collect_predictions(model: nn.Module, loader: DataLoader) -> tuple[list[int], list[int]]:
    """Collect labels and predictions for confusion-matrix evaluation."""
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for images, batch_labels in loader:
            predictions.extend(model(images).argmax(dim=1).tolist())
            labels.extend(batch_labels.tolist())
    return labels, predictions


def build_confusion_matrix(labels: list[int], predictions: list[int], class_count: int) -> list[list[int]]:
    """Build a confusion matrix without a tabular ML dependency."""
    matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for label, prediction in zip(labels, predictions):
        matrix[label][prediction] += 1
    return matrix


def train_model(
    data_dir: Path = DEFAULT_DATA_DIR, epochs: int = 5, batch_size: int = 32, learning_rate: float = 1e-3, seed: int = 42
) -> TrainingResult:
    """Train the CNN, log the run to MLflow, and save its state dictionary."""
    torch.manual_seed(seed)
    train_loader, validation_loader, test_loader, classes = build_dataloaders(data_dir, batch_size, seed)
    model = BaselineCNN(num_classes=len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="baseline-cnn"):
        mlflow.log_params({"model": "BaselineCNN", "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "seed": seed, "image_size": "224x224"})
        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer)
            validation_loss, validation_accuracy = run_epoch(model, validation_loader, criterion)
            mlflow.log_metrics({"train_loss": train_loss, "train_accuracy": train_accuracy, "validation_loss": validation_loss, "validation_accuracy": validation_accuracy}, step=epoch)

            print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | Val Loss: {validation_loss:.4f}, Val Acc: {validation_accuracy:.4f}")
            
        test_loss, test_accuracy = run_epoch(model, test_loader, criterion)
        labels, predictions = collect_predictions(model, test_loader)
        matrix = build_confusion_matrix(labels, predictions, len(classes))
        figure, axis = plt.subplots()
        axis.imshow(matrix, cmap="Blues")
        axis.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes, yticklabels=classes, xlabel="Predicted label", ylabel="True label")
        for row in range(len(classes)):
            for column in range(len(classes)):
                axis.text(column, row, matrix[row][column], ha="center", va="center")
        figure.tight_layout()
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        confusion_path = ARTIFACT_DIR / "confusion_matrix.png"
        figure.savefig(confusion_path)
        plt.close(figure)
        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_accuracy})
        mlflow.log_artifact(str(confusion_path), artifact_path="evaluation")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "classes": classes}, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
    result = TrainingResult(str(MODEL_PATH), classes, test_loss, test_accuracy)
    (ARTIFACT_DIR / "training_summary.json").write_text(json.dumps(asdict(result), indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Cats vs Dogs baseline CNN.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    print(json.dumps(asdict(train_model(**vars(args))), indent=2))


if __name__ == "__main__":
    main()
    