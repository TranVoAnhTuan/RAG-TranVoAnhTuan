FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    POETRY_VERSION=2.3.0 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "poetry==$POETRY_VERSION"

WORKDIR /app

# Install only the [ui] Poetry group
COPY pyproject.toml poetry.lock ./
RUN poetry install --only ui --no-root --no-cache

# Copy UI source
COPY ui/ ./ui/

EXPOSE 8501

# BACKEND_URL is overridden by docker-compose environment
ENV BACKEND_URL=http://backend:8000

CMD ["streamlit", "run", "ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
