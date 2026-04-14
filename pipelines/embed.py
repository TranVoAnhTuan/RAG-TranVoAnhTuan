import uuid
import asyncio
import gc
import torch
from qdrant_client import models
from qdrant_client.models import PointStruct, SparseVector
from app.db.qdrant_db import QdrantDatabase
from app.core.config import settings
from .state import IngestionState
from app.core.model_manager import model_manager
import logging

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def embed_and_load_node(state: IngestionState):
    chunks = state["chunks"]
    minio_url = state.get("minio_url")
    
    logger.info(f"--- [CHECKPOINT 1] Started embed_and_load_node. Total chunks received: {len(chunks)} ---")
    
    try:       
        logger.info("--- [CHECKPOINT 2] Connecting to Qdrant and configuring... ---")
        db = QdrantDatabase()
        client = db.get_client()
        await db._ensure_collection_exists()

        await client.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            field_name="Header_1",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        await db.enable_fast_upload_mode()
        logger.info("--- [CHECKPOINT 2] Qdrant configured (fast_upload_mode: ON). ---")

        texts = [item["content"] for item in chunks if item["content"]]
        
        if texts:
            logger.info(f"--- [CHECKPOINT 3] Generating Sparse embeddings for {len(texts)} texts... ---")
            sparse_model = model_manager.get_sparse_model()
            sparse_embeddings = list(sparse_model.embed(texts))
            logger.info("--- [CHECKPOINT 3] Sparse embeddings complete. ---")
            
            logger.info("--- [CHECKPOINT 4] Borrowing GPU for Dense embeddings... ---")
            with model_manager.use_dense_model_on_gpu() as gpu_dense_model:
                dense_embeddings = gpu_dense_model.encode(texts, batch_size=2, normalize_embeddings=True)
            logger.info("--- [CHECKPOINT 4] Dense embeddings complete, GPU released. ---")

            logger.info("--- [CHECKPOINT 5] Assembling PointStructs... ---")
            points = []
            for i, item in enumerate([c for c in chunks if c["content"]]):
                dense_vec = dense_embeddings[i].tolist()
                sparse_dict = sparse_embeddings[i]
                if sparse_dict and len(sparse_dict.indices) > 0:
                    indices = sparse_dict.indices.tolist() if hasattr(sparse_dict.indices, 'tolist') else list(sparse_dict.indices)
                    values = sparse_dict.values.tolist() if hasattr(sparse_dict.values, 'tolist') else list(sparse_dict.values)
                    
                    sparse_vec = SparseVector(
                        indices=indices,
                        values=values
                    )
                else:
                    sparse_vec = SparseVector(indices=[], values=[])
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": dense_vec,
                        "sparse": sparse_vec
                    },
                    payload={
                        "content": item["content"],
                        "Header_1": item.get("Header_1"),
                        "Header_2": item.get("Header_2"),
                        "file_url": minio_url   
                    }
                )
                points.append(point)
            logger.info(f"--- [CHECKPOINT 6] Successfully created {len(points)} points. ---")

            if points:
                batch_size = 100
                tasks = []  
                for i in range(0, len(points), batch_size):
                    batch = points[i : i + batch_size]
                    tasks.append(client.upsert(
                        collection_name=settings.QDRANT_COLLECTION_NAME, 
                        points=batch
                    ))
                
                logger.info(f"--- [CHECKPOINT 6] Uploading to Qdrant in {len(tasks)} batches (Batch size: {batch_size} points)... ---")
                
                # ANTI-FREEZE TRICK: Avoid sending all tasks at once. Chunk the gather process.
                chunked_tasks_size = 10 
                for j in range(0, len(tasks), chunked_tasks_size):
                    task_chunk = tasks[j : j + chunked_tasks_size]
                    await asyncio.gather(*task_chunk)
                    logger.info(f"   -> Uploaded batch {j} to {j + len(task_chunk)} / {len(tasks)}")
                
                logger.info("--- [CHECKPOINT 7] ALL DATA SUCCESSFULLY UPLOADED TO QDRANT. ---")
                
        logger.info("--- [CHECKPOINT 8] Re-enabling HNSW Search mode (This might take a moment)... ---")
        await db.enable_fast_search_mode()
        logger.info("--- [CHECKPOINT 8] Re-enabled HNSW indexing successfully. ---")

        return {"status": f"The document was processed, and {len(texts)} chunks were saved to Qdrant successfully!"}

    except Exception as e:
        logger.exception("CRITICAL ERROR IN EMBED NODE") 
        return {"status": f"Failed with error: {str(e)}"}