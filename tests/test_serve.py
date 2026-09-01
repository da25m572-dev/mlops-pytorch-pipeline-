"""Unit tests for src/serve.py (FastAPI serving app)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi.testclient import TestClient  # noqa: E402
from serve import app  # noqa: E402


def test_health_returns_503_when_model_not_loaded() -> None:
    # No checkpoint exists at the default path in the test environment, so the
    # app should report unhealthy rather than crash.
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["model_loaded"] is False


def test_predict_returns_503_when_model_not_loaded() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"image": ("test.png", b"not-a-real-image", "image/png")},
        )
    assert response.status_code == 503
