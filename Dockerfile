# ── FastAPI Backend ────────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    POETRY_VERSION=2.3.0 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "poetry==$POETRY_VERSION"

WORKDIR /app

# ── Install main dependencies (mcp + ui groups are optional=true, skipped) ─────
COPY pyproject.toml poetry.lock ./
RUN poetry install --without mcp,ui --no-root --no-cache

# ── Copy application source ────────────────────────────────────────────────────
COPY app/ ./app/
COPY pipelines/ ./pipelines/

# ── Optional: SSL certs for Qdrant/MongoDB TLS ────────────────────────────────
# COPY combined.pem ./combined.pem

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
