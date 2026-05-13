"""Pre-download all ML models to the local cache.

Can be run standalone (``python -m scripts.download_models``) or called
from application startup via :func:`ensure_models_available`.

Cache directories are managed by the upstream libraries
(``huggingface_hub`` and ``fastembed``) and are automatically
cross-platform (Linux, macOS, Windows).
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Read model names from environment, falling back to the same defaults
# used by app/core/config.py and mcp_server/config.py.
DENSE_MODEL_NAME = os.getenv("DENSE_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")
SPARSE_MODEL_NAME = os.getenv("SPARSE_MODEL_NAME", "Qdrant/bm42-all-minilm-l6-v2-attentions")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "jinaai/jina-reranker-v3")


def ensure_models_available() -> None:
    """Check and download all required models if not already cached.

    Uses ``huggingface_hub.snapshot_download`` for HuggingFace models
    and ``fastembed.SparseTextEmbedding`` for the sparse model.
    Both libraries resolve cache paths automatically on every OS.
    """
    from huggingface_hub import snapshot_download

    # ── 1. Dense embedding model ──────────────────────────────────────
    logger.info(f"📦 Checking Dense model: {DENSE_MODEL_NAME} …")
    snapshot_download(DENSE_MODEL_NAME)
    logger.info(f"✅ Dense model ready: {DENSE_MODEL_NAME}")

    # ── 2. Reranker model ─────────────────────────────────────────────
    logger.info(f"📦 Checking Reranker model: {RERANKER_MODEL_NAME} …")
    snapshot_download(RERANKER_MODEL_NAME)
    logger.info(f"✅ Reranker model ready: {RERANKER_MODEL_NAME}")

    # ── 3. Sparse embedding model (fastembed) ─────────────────────────
    logger.info(f"📦 Checking Sparse model: {SPARSE_MODEL_NAME} …")
    from fastembed import SparseTextEmbedding

    # fastembed downloads on instantiation; discard the object immediately.
    SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
    logger.info(f"✅ Sparse model ready: {SPARSE_MODEL_NAME}")

    logger.info("🎉 All models are available.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    ensure_models_available()
