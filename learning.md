# B1: HuggingFace Fine-Tuning + Production FastAPI
# Owner: Priyrajsinh Parmar | github.com/Priyrajsinh
Stack: Python 3.12, transformers, datasets, torch, mlflow, fastapi, uvicorn,
gradio, pandera, pydantic, slowapi, prometheus-fastapi-instrumentator,
python-json-logger, bandit, scikit-learn, pandas, numpy, psutil
Config: ALL hyperparameters in config/config.yaml — never hardcode
Logging: get_logger from src/logger.py — JSON-aware (ENV=production)
Exceptions: all errors through src/exceptions.py (PredictionError, DataLoadError)
Base class: SentimentClassifier implements src/models/base.py BaseMLModel
Validation: pandera EMOTION_SCHEMA in src/data/validation.py before any split
Schemas: SentimentInput / SentimentOutput Pydantic in src/data/schemas.py
Security: bandit -r src/ -ll must return zero findings
Monitoring: /metrics (Prometheus), /api/v1/health (model_loaded+uptime+memory)
Dataset: GoEmotions mapped to 7 macro-categories
Model: distilbert-base-uncased fine-tuned via HuggingFace Trainer API
Labels: joy=0, sadness=1, anger=2, fear=3, surprise=4, disgust=5, neutral=6
NEVER commit: models/sentiment_model/ (~250MB)
make install / train / test / lint / serve / docker-build / gradio / audit
