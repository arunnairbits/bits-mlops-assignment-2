# Cats vs Dogs MLOps Pipeline

End-to-end computer vision MLOps pipeline for binary cats-versus-dogs image classification. The project uses PyTorch, DVC, MLflow, FastAPI, Docker, Prometheus, and Kubernetes.

## Prerequisites

- Python 3.12+
- `uv`
- Docker Desktop with Compose
- `kubectl` and a local Kubernetes cluster for Kubernetes deployment
- Kaggle API credentials for downloading the dataset

## Setup

### Install uv

On macOS, install `uv` with Homebrew:

```bash
brew install uv
```

Alternatively, use the official macOS/Linux installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows, run one of the following in PowerShell:

```powershell
winget install --id=astral-sh.uv -e
```

Or use the official PowerShell installer:

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Restart the terminal if needed, then verify the installation:

```bash
uv --version
```

Install the locked Python dependencies:

```bash
uv sync
```

Create a Kaggle API token in your Kaggle account settings, then copy the sample environment file and replace the placeholder with your token. The `.env` file is ignored by Git.

On macOS/Linux:

```bash
cp .env.sample .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.sample .env
```

Set `KAGGLE_API_TOKEN` in `.env`. `KAGGLE_DATASET` is optional and defaults to `bhavikjikadara/dog-and-cat-classification-dataset`.

Then download and version the dataset:

```bash
uv run python -m src.acquire_data
dvc add data/raw
```

Generate the resized `224x224` RGB dataset and track it with DVC:

```bash
uv run python -m scripts.generate_processed_data
dvc add data/processed
```

## Train And Track Experiments

Start the local MLflow tracking server before training:

```bash
uv run mlflow server --backend-store-uri sqlite:////$(pwd)/mlruns/mlflow.db --serve-artifacts --host 127.0.0.1 --port 5000
```

Then train the baseline CNN. This writes the model and training summary to `artifacts/` and logs metrics and evaluation artifacts to MLflow:

```bash
MLFLOW_TRACKING_URI="http://localhost:5000" \
	uv run python -m src.train --data-dir data/raw/PetImages --epochs 5
```

## Tests And Lint

```bash
uv run pytest -q
uv run ruff check .
```

## Docker Compose

Build and start the API, Prometheus, and MLflow services:

```bash
docker compose build api
docker compose up -d
```

The Compose API is available at `http://localhost:8001`; container port `8000` is mapped to host port `8001`.

```bash
curl http://localhost:8001/health
curl -X POST -F "file=@path/to/image.jpg" http://localhost:8001/predict
```

Prometheus is available at `http://localhost:9095` and MLflow at `http://localhost:5000`.

To stop the stack:

```bash
docker compose down
```

## Kubernetes

Apply the manifests to the active local cluster:

```bash
kubectl apply -f k8s/
kubectl port-forward service/cats-dogs-api-service 8000:8000
```

The Kubernetes manifest currently references `localhost:5001/cats-dogs-api:local`; make that image available to the cluster before applying it. With port-forwarding active:

```bash
curl http://localhost:8000/health
curl -X POST -F "file=@path/to/image.jpg" http://localhost:8000/predict
```

## Post-deployment Evaluation

Evaluate labeled images from the processed dataset against a running API:

```bash
uv run python scripts/post_deploy_eval.py \
	--api-url http://localhost:8001/predict \
	--data-dir data/processed \
	--samples-per-class 20
```

The script reports request count, average latency, P95 latency, accuracy, and the confusion matrix.