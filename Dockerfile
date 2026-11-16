# Retail Repeat-Purchase Prediction Service — container image
# Builds a lean API image that serves the pre-trained model artifact.
#
# Prerequisite: models/model.joblib must exist (python scripts/train.py).
# Build:  docker build -t retail-repeat:latest .
# Run:    docker run --rm -p 8000:8000 retail-repeat:latest
# Docs:   http://127.0.0.1:8000/docs
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install runtime dependencies first (cached layer unless requirements change)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Application code + the trained model artifact
COPY src ./src
COPY api ./api
COPY models/model.joblib ./models/model.joblib

# Run as an unprivileged user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Container-level liveness check hitting the app's own /health endpoint
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
