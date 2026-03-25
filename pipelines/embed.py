import uuid
import asyncio
import gc
import torch
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client import models
from qdrant_client.models import PointStruct, SparseVector
from app.db.qdrant_db import QdrantDatabase
from app.core.config import settings
from .state import IngestionState
from app.core.model_manager import model_manager

async def embed_and_load_node(state: IngestionState):
    chunks = state["chunks"]
    
    print("Loading Embedding Models...")
    sparse_model = model_manager.get_sparse_model()
    dense_model = model_manager.get_dense_model()
    # sparse_model = SparseTextEmbedding(model_name=settings.SPARSE_MODEL_NAME)
    # dense_model = SentenceTransformer(settings.DENSE_MODEL_NAME, trust_remote_code=True)
    
    db = QdrantDatabase()
    client = db.get_client()
    await db._ensure_collection_exists()

    await client.create_payload_index(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        field_name="Header_1",
        field_schema=models.PayloadSchemaType.KEYWORD
    )

    # Gom texts để encode batch cho nhanh
    texts = [item["content"] for item in chunks if item["content"]]
    
    if texts:
        print(f"Encoding {len(texts)} chunks...")
        # Dense
        dense_embeddings = dense_model.encode(texts, batch_size=32, normalize_embeddings=True)
        # Sparse
        sparse_embeddings = list(sparse_model.embed(texts))

        points = []
        for i, item in enumerate([c for c in chunks if c["content"]]):
            dense_vec = dense_embeddings[i].tolist()
            sparse_vec = sparse_embeddings[i]
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vec,
                    "sparse": SparseVector(indices=sparse_vec.indices.tolist(), values=sparse_vec.values.tolist())
                },
                payload={
                    "content": item["content"],
                    "Header_1": item.get("Header_1"),
                    "Header_2": item.get("Header_2")
                }
            )
            points.append(point)

        if points:
            batch_size = 10  
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                await client.upsert(collection_name=settings.QDRANT_COLLECTION_NAME, points=batch)
                await asyncio.sleep(0.1) 

    # Dọn dẹp RAM / VRAM cực mạnh
    print("Cleaning Embedding Models RAM/VRAM...")
    # del dense_model
    # del sparse_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"status": f"The document was processed, and {len(texts)} chunks were saved to Qdrant successfully!"}