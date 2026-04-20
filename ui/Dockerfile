FROM python:3.11-slim

# ── Certificates for Corporate Proxy / Internet Block ─────────────────────────
USER root
COPY combined.pem /usr/local/share/ca-certificates/combined.crt
RUN apt-get update && apt-get install -y ca-certificates && update-ca-certificates

ENV DEBIAN_FRONTEND=noninteractive \
    POETRY_VERSION=2.3.0 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "poetry==$POETRY_VERSION"

WORKDIR /app

# Install only the [ui] Poetry group
COPY pyproject.toml poetry.lock ./
RUN poetry lock && \
    poetry install --only ui --no-root --no-cache

# Copy UI source
COPY ui/ ./ui/

# Keep a copy of combined.pem in working directory just in case
COPY combined.pem ./combined.pem

EXPOSE 8501

# BACKEND_URL is overridden by docker-compose environment
ENV BACKEND_URL=http://backend:8000

CMD ["streamlit", "run", "ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
