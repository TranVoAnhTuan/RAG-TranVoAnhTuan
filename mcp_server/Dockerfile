# ── Base image: CUDA + Python ──────────────────────────────────────────────────
# Adjust the CUDA / cuDNN version to match your host driver.
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# ── System dependencies ────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    ca-certificates \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install pip explicitly for python3.11 to avoid ubuntu 22.04's default python3.10 pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Make python3.11 the default python / pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.11 1

# ── Install Poetry ─────────────────────────────────────────────────────────────
ENV POETRY_VERSION=2.3.0 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1
RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Install only the [mcp] dependency group ────────────────────────────────────
# Copy lock files first to maximise Docker layer caching.
# Changing source code won't invalidate this layer.
COPY pyproject.toml poetry.lock ./
RUN poetry install --only mcp --no-root --no-cache

# ── Copy MCP server source code ────────────────────────────────────────────────
COPY mcp_server/ ./mcp_server/

# ── Optional: copy SSL certs if your Qdrant / MinIO endpoint needs them ────────
# COPY combined.pem /app/combined.pem
# ENV SSL_CERT_FILE=/app/combined.pem
# ENV REQUESTS_CA_BUNDLE=/app/combined.pem

# ── Expose the MCP HTTP port ───────────────────────────────────────────────────
EXPOSE 8001

# ── Start the FastMCP server ───────────────────────────────────────────────────
CMD ["python", "-m", "mcp_server.server"]
