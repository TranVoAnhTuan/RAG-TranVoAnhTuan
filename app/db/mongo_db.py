"""MongoDB client for the FastAPI backend.

Singleton pattern: use the module-level ``mongo_db`` instance.
"""

from pymongo import MongoClient

from app.core.config import settings


class MongoDB:
    """Provides a shared MongoDB connection and default database/collection handles."""

    def __init__(self) -> None:
        self.client: MongoClient = MongoClient(settings.MONGODB_HOST)
        self.db = self.client["agentic_rag_database"]
        self.collection = self.db["documents_metadata"]


mongo_db = MongoDB()
