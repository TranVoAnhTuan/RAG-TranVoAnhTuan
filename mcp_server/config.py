"""MCP server configuration loaded from environment variables.

Singleton pattern: use the module-level ``mcp_settings`` instance.
"""

import os

from dotenv import load_dotenv

load_dotenv()


class MCPSettings:
    """Configuration for the standalone MCP tools server (separate Docker container)."""

    # ── Qdrant ──────────────────────────────────────────────────────
    QDRANT_URL: str | None = os.getenv("QDRANT_URL")
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "intern_rag_agent")

    # ── Embedding / Reranker models ─────────────────────────────────
    SPARSE_MODEL_NAME: str = os.getenv("SPARSE_MODEL_NAME", "Qdrant/bm42-all-minilm-l6-v2-attentions")
    DENSE_MODEL_NAME: str = os.getenv("DENSE_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")
    RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME", "jinaai/jina-reranker-v3")
    CONTEXT_LENGTH: int = int(os.getenv("CONTEXT_LENGTH", 8000))

    # ── MinIO public endpoint (for presigned URLs shown to browser) ──
    MINIO_PUBLIC_ENDPOINT: str = os.getenv("MINIO_PUBLIC_ENDPOINT", "localhost:9000")

    # ── Auth / Keys ─────────────────────────────────────────────────
    HUGGINGFACE_ACCESS_TOKEN: str | None = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
    TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")


mcp_settings = MCPSettings()
