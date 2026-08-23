"""FastAPI service exposing Cats vs Dogs image classification."""

from __future__ import annotations

import logging
import time
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from prometheus_fastapi_instrumentator import Instrumentator

from src.predict import MODEL_PATH, load_model, predict_image

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api_logger")

app = FastAPI(title="Cats vs Dogs Predictor", version="1.0.0")
model = None
classes: list[str] = []
Instrumentator().instrument(app).expose(app)


@app.middleware("http")
async def log_requests(request, call_next):
    """Log request method, path, status, and latency."""
    start_time = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start_time
    logger.info("Method: %s Path: %s Status: %s Latency: %.4fs", request.method, request.url.path, response.status_code, latency)
    return response


@app.on_event("startup")
def startup_event() -> None:
    """Load the trained model when the service starts."""
    global model, classes
    if MODEL_PATH.exists():
        model, classes = load_model(MODEL_PATH)
        logger.info("CNN model loaded successfully from %s", MODEL_PATH)
    else:
        logger.warning("Model artifact not found at %s", MODEL_PATH)


@app.get("/health")
def health() -> dict[str, object]:
    """Return the service readiness status."""
    return {"status": "ok", "model_ready": model is not None}


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File()]) -> dict[str, str | int | float]:
    """Classify one uploaded image and return its class confidence."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")
    try:
        result = predict_image(await file.read(), model, classes)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    logger.info("Prediction generated: %s (confidence: %.4f)", result["label"], result["confidence"])
    return result