# ── FastAPI Backend (GPU Enabled) ─────────────────────────────────────────────
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# ── Certificates for Corporate Proxy / Internet Block ─────────────────────────
USER root
COPY combined.pem /usr/local/share/ca-certificates/combined.crt
RUN update-ca-certificates

ENV DEBIAN_FRONTEND=noninteractive \
    POETRY_VERSION=2.3.0 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cu13/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
    libxcb1 libxrender1 libxext6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "poetry==$POETRY_VERSION"

WORKDIR /app

# ── Install main dependencies ──────────────────────────────────────────────────
COPY pyproject.toml poetry.lock ./
RUN poetry install --without mcp,ui --no-root --no-cache

# ── Copy application source ────────────────────────────────────────────────────
COPY app/ ./app/
COPY pipelines/ ./pipelines/

# Keep a copy of combined.pem in working directory just in case
COPY combined.pem ./combined.pem

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
