import os
import json
from dotenv import load_dotenv

load_dotenv()


class MCPSettings:
    # Qdrant
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "intern_rag_agent")

    # Embedding / Reranker models
    SPARSE_MODEL_NAME = os.getenv(
        "SPARSE_MODEL_NAME", "Qdrant/bm42-all-minilm-l6-v2-attentions"
    )
    DENSE_MODEL_NAME = os.getenv("DENSE_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")
    RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "jinaai/jina-reranker-v3")
    CONTEXT_LENGTH = int(os.getenv("CONTEXT_LENGTH", 8000))

    HUGGINGFACE_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")


mcp_settings = MCPSettings()
