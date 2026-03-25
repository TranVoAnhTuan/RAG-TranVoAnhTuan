from qdrant_client import AsyncQdrantClient # Sử dụng bản async
from qdrant_client.models import VectorParams, Distance, SparseVectorParams
from app.core.config import settings

class QdrantDatabase:
    def __init__(self):
        self.client = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout= 120
        )
        self.collection_name = settings.QDRANT_COLLECTION_NAME

    async def _ensure_collection_exists(self):
        exists = await self.client.collection_exists(self.collection_name)
        if not exists:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=1024, distance=Distance.COSINE) 
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams()
                }
            )

    def get_client(self) -> AsyncQdrantClient:
        return self.client