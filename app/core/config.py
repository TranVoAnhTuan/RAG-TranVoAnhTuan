"""Application settings loaded from environment variables.

Singleton pattern: use the module-level ``settings`` instance.
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class Settings:
    """Central configuration for the FastAPI backend.

    All values are read from environment variables at import time.
    Access via the module-level singleton ``settings``.
    """

    # ── API / Auth ──────────────────────────────────────────────────
    API_KEY: str | None = os.getenv("API_KEY")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    # ── LLM ─────────────────────────────────────────────────────────
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen3.5:2b")
    BASE_URL: str | None = os.getenv("BASE_URL")
    GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")

    # ── Qdrant ──────────────────────────────────────────────────────
    QDRANT_URL: str | None = os.getenv("QDRANT_URL")
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "intern_rag_agent")

    # ── SQLite / MinIO ──────────────────────────────────────────────
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "rag_database.db")
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "rag-agent")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "89898989")
    MINIO_BUCKET_NAME: str = os.getenv("MINIO_BUCKET_NAME", "pdfs")

    # ── MongoDB ─────────────────────────────────────────────────────
    MONGODB_HOST: str | None = os.getenv("DATABASE_HOST")

    # ── HuggingFace ─────────────────────────────────────────────────
    HUGGINGFACE_ACCESS_TOKEN: str | None = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

    # ── MCP Server URL ──────────────────────────────────────────────
    # In Docker: http://mcp-server:8001/mcp  |  Local dev: http://localhost:8001/mcp
    MCP_SERVER_URL: str = os.getenv("MCP_SERVER_URL", "http://localhost:8001/mcp")

    # ── Embedding / Reranker Models ─────────────────────────────────
    SPARSE_MODEL_NAME: str = os.getenv("SPARSE_MODEL_NAME", "Qdrant/bm42-all-minilm-l6-v2-attentions")
    DENSE_MODEL_NAME: str = os.getenv("DENSE_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")
    CONTEXT_LENGTH: int = int(os.getenv("CONTEXT_LENGTH", 8000))
    RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME", "jinaai/jina-reranker-v3")

    # ── Cache ───────────────────────────────────────────────────────
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", 86400))


settings = Settings()
