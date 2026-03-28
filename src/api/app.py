"""Production FastAPI application for the Distilbert Emotion Classifier.

Full Production v5 Pattern:
- SlowAPI rate limiting (30/minute on /predict)
- CORS and TrustedHost middleware
- Content-Length guard (reject > 1 MB)
- Prometheus metrics via prometheus-fastapi-instrumentator
- Pandera validation before inference
- PredictionError exception handler -> HTTP 422
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pandera.pandas as pa
import psutil
import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.data.schemas import SentimentInput, SentimentOutput
from src.exceptions import PredictionError
from src.logger import get_logger
from src.models.model import SentimentClassifier

logger = get_logger(__name__)

_CONFIG_PATH = Path("config/config.yaml")
_RESULTS_PATH = Path("reports/results.json")


def _load_config() -> dict:
    with _CONFIG_PATH.open() as fh:
        return yaml.safe_load(fh)


_config = _load_config()

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Distilbert Emotion Classifier API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# Middleware (exact order from spec)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=_config["api"]["cors_origins"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*"],
)

_MAX_PAYLOAD_BYTES: int = int(_config["api"]["max_payload_mb"] * 1024 * 1024)


@app.middleware("http")
async def _check_content_length(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > _MAX_PAYLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={
                "detail": (
                    f"Payload too large — max {_config['api']['max_payload_mb']} MB"
                )
            },
        )
    return await call_next(request)


Instrumentator().instrument(app).expose(app)


@app.exception_handler(PredictionError)
async def _prediction_error_handler(request: Request, exc: PredictionError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# Pandera schema for single-row inference input
# ---------------------------------------------------------------------------
_PREDICT_SCHEMA = pa.DataFrameSchema(
    {"text": pa.Column(str, nullable=False)},
    strict=True,
)

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------
classifier: SentimentClassifier | None = None
model_loaded: bool = False
START_TIME: float = 0.0


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _startup() -> None:
    global classifier, model_loaded, START_TIME
    START_TIME = time.time()
    try:
        classifier = SentimentClassifier()
        classifier.load(Path("models/sentiment_model"))
        model_loaded = True
        logger.info("Model loaded successfully")
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/health")
async def health():
    proc = psutil.Process()
    memory_mb = proc.memory_info().rss / (1024 * 1024)
    return {
        "status": "ok" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "memory_mb": round(memory_mb, 2),
    }


@app.post("/api/v1/predict", response_model=SentimentOutput)
@limiter.limit("30/minute")
async def predict(request: Request, body: SentimentInput) -> SentimentOutput:
    df = pd.DataFrame([{"text": body.text}])
    _PREDICT_SCHEMA.validate(df)

    proba_dict = classifier.predict_proba([body.text])[0]  # type: ignore[union-attr]
    label = max(proba_dict, key=lambda k: proba_dict[k])
    confidence = proba_dict[label]

    return SentimentOutput(
        label=label,
        confidence=confidence,
        probabilities=proba_dict,
        trace_id=str(uuid4()),
    )


@app.get("/api/v1/model_info")
async def model_info():
    cfg = _config["model"]
    evaluation = None
    if _RESULTS_PATH.exists():
        with _RESULTS_PATH.open() as fh:
            evaluation = json.load(fh)
    return {
        "model": cfg["base_model"],
        "num_labels": cfg["num_labels"],
        "evaluation": evaluation,
    }
