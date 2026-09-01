"""FastAPI inference service for the mlops-pytorch-pipeline image classifier.

Loads a trained checkpoint on startup and exposes:
  - GET  /health  -> 200 if the model is loaded, 503 otherwise
  - POST /predict -> accepts an uploaded image, returns class probabilities
"""
from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from dataset import get_transforms
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from model import get_model
from PIL import Image

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_transform = get_transforms(train=False)
_state: dict = {"model": None}


def resolve_checkpoint_path() -> Path:
    """Resolve the model checkpoint path.

    Priority: ``CHECKPOINT_PATH`` env var (full file path) > ``CHECKPOINT_DIR`` +
    ``MODEL_NAME`` env vars > container default ``/app/checkpoints/classifier_v1.pt``.
    """
    env_path = os.environ.get("CHECKPOINT_PATH")
    if env_path:
        return Path(env_path)
    checkpoint_dir = Path(os.environ.get("CHECKPOINT_DIR", "/app/checkpoints"))
    model_name = os.environ.get("MODEL_NAME", "classifier_v1.pt")
    return checkpoint_dir / model_name


def load_model() -> torch.nn.Module | None:
    """Load the model checkpoint from disk, if present."""
    checkpoint_path = resolve_checkpoint_path()
    if not checkpoint_path.exists():
        return None

    checkpoint = torch.load(checkpoint_path, map_location=_device, weights_only=False)
    model = get_model(
        architecture=checkpoint.get("architecture", "resnet18"),
        num_classes=checkpoint.get("num_classes", 10),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(_device)
    model.eval()
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["model"] = load_model()
    yield
    _state["model"] = None


app = FastAPI(title="mlops-pytorch-pipeline serving", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> JSONResponse:
    """Liveness/readiness probe target. 200 if the model is loaded, 503 otherwise."""
    if _state["model"] is None:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "model_loaded": False})
    return JSONResponse(status_code=200, content={"status": "healthy", "model_loaded": True})


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict:
    """Run inference on an uploaded image and return class probabilities."""
    model = _state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    pil_image = pil_image.resize((32, 32))
    tensor = _transform(pil_image).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0)

    predicted_idx = int(probabilities.argmax().item())
    return {
        "predicted_class": CIFAR10_CLASSES[predicted_idx],
        "predicted_index": predicted_idx,
        "probabilities": {
            CIFAR10_CLASSES[i]: round(p, 6) for i, p in enumerate(probabilities.tolist())
        },
    }

