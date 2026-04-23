import os
from dotenv import load_dotenv
import json
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class Settings:
    API_KEY = os.getenv("API_KEY")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

    LLM_MODEL = os.getenv("LLM_MODEL","qwen3.5:2b")
    BASE_URL = os.getenv("BASE_URL")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "intern_rag_agent")
    SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "rag_database.db")
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "rag-agent")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "89898989")
    MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "pdfs")
    MONGODB_HOST = os.getenv("DATABASE_HOST")
    
    HUGGINGFACE_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

    # MCP Server URL — used by DemoAgent (MCP client) to connect to the tools service.
    # In Docker: http://mcp-server:8001/mcp  |  Local dev: http://localhost:8001/mcp
    MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001/mcp")

    SPARSE_MODEL_NAME = os.getenv("SPARSE_MODEL_NAME", "Qdrant/bm42-all-minilm-l6-v2-attentions")
    DENSE_MODEL_NAME = os.getenv("DENSE_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")
    CONTEXT_LENGTH = int(os.getenv("CONTEXT_LENGTH", 8000))

    RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "jinaai/jina-reranker-v3")

    _default_topics = {
        "Insurance": "Includes regulations, policies, company information, and guidelines related to the insurance industry.",
        "QNU": "Information related to Quy Nhon University (QNU), including university regulations, academics, admissions, and campus rules.",
        "General": "General topics or documents that do not strictly fit into Insurance or QNU categories."
    }

    _topics_env = os.getenv("AVAILABLE_TOPICS_JSON")
    if _topics_env:
        try:
            AVAILABLE_TOPICS = json.loads(_topics_env)
        except json.JSONDecodeError:
            logger.warning("⚠️ Lỗi parse AVAILABLE_TOPICS_JSON trong .env, sử dụng topics mặc định.")
            AVAILABLE_TOPICS = _default_topics
    else:
        AVAILABLE_TOPICS = _default_topics

settings = Settings()