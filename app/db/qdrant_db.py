"""Async Qdrant vector database client.

Singleton pattern: instantiate via ``QdrantDatabase()`` and call
``get_client()`` to obtain the underlying ``AsyncQdrantClient``.
"""

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, HnswConfigDiff, SparseVectorParams, VectorParams

from app.core.config import settings


class QdrantDatabase:
    """Manages an async Qdrant connection and collection lifecycle."""

    def __init__(self) -> None:
        self.client = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=120,
        )
        self.collection_name: str = settings.QDRANT_COLLECTION_NAME

    async def _ensure_collection_exists(self) -> None:
        """Create the collection with dense + sparse vectors if it doesn't exist."""
        exists = await self.client.collection_exists(self.collection_name)
        if not exists:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams()},
                hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
                shard_number=2,
            )

    async def enable_fast_upload_mode(self) -> None:
        """Disable HNSW indexing for bulk uploads."""
        await self.client.update_collection(
            collection_name=self.collection_name,
            hnsw_config=HnswConfigDiff(m=0),
        )

    async def enable_fast_search_mode(self) -> None:
        """Re-enable HNSW indexing for optimised search."""
        await self.client.update_collection(
            collection_name=self.collection_name,
            hnsw_config=HnswConfigDiff(m=32, ef_construct=100),
        )

    def get_client(self) -> AsyncQdrantClient:
        """Return the underlying async client."""
        return self.client
