from qdrant_client import models
from app.db.qdrant_db import QdrantDatabase
from app.core.config import settings
import asyncio
from app.core.model_manager import model_manager

class RAGService:
    def __init__(self):
        self.db = QdrantDatabase()
        self.client = self.db.get_client()
        
       # NOTE 1: Only keep the Sparse model as it runs on CPU.
        # DO NOT initialize Dense Model and Reranker here to avoid occupying VRAM permanently.
        self.sparse_model = model_manager.get_sparse_model()

    async def retrieve_and_rerank(self, query_text: str) -> str:
        
        # 1. Embed query
        loop = asyncio.get_event_loop()
        def get_embeddings():
            # NOTE 2: Borrow GPU to create Dense Vector, release immediately after
            with model_manager.use_dense_model_on_gpu() as gpu_dense_model:
                dense = gpu_dense_model.encode(query_text).tolist() 
            
            sparse = list(self.sparse_model.embed([query_text]))[0]
            return dense, sparse

        dense_query, sparse_query = await loop.run_in_executor(None, get_embeddings)

        sparse_query_vector = models.SparseVector(
            indices=list(sparse_query.indices) if sparse_query.indices is not None else [],
            values=list(sparse_query.values) if sparse_query.values is not None else []
        )

        # 2. Hybrid Search (This process calls DB API, VRAM is 100% free at this time)
        hybrid_results = await self.client.query_points(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_query,
                    using="dense",
                    limit=20,
                ),
                models.Prefetch(
                    query=sparse_query_vector,
                    using="sparse",
                    limit=20,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=10,
        )

        if not hybrid_results.points:
            return "No appropriate information was found."

        merged_docs_dict = {doc.id: doc for doc in hybrid_results.points}
        processed_headers = {doc.payload.get("Header_1") for doc in hybrid_results.points if doc.payload.get("Header_1")}

        for header in processed_headers:
            try:
                filter_points, _ = await self.client.scroll(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    scroll_filter=models.Filter(
                        must=[models.FieldCondition(key="Header_1", match=models.MatchValue(value=header))]
                    ),
                    limit=10,
                    with_payload=True,
                )
                for p in filter_points:
                    merged_docs_dict[p.id] = p
            except Exception:
                continue

        merged_docs = list(merged_docs_dict.values())

        # 3. Rerank
        def do_rerank():
            documents = [doc.payload.get('content', '') for doc in merged_docs]
            
            # NOTE 3: Borrow GPU to run Reranker, release immediately after scoring
            with model_manager.use_reranker_model() as gpu_reranker:
                scores= gpu_reranker.rerank(query_text, documents)
            
                # del gpu_reranker 
            return scores

        jina_results = await loop.run_in_executor(None, do_rerank)
        
        # Get the top 5 best results. 
        top_docs = []
        for result in jina_results[:5]:
            original_idx = result['index']
            top_docs.append(merged_docs[original_idx])

        search_result = "SEARCH RESULTS:\n"
        for i, doc in enumerate(top_docs):
            h1 = doc.payload.get('Header_1', '')
            h2 = doc.payload.get('Header_2', '')
            f_url = doc.payload.get('file_url', '')
            content = doc.payload.get('content', '')
            
            search_result += f"""
--- Result {i+1} ---
CONTENT: {content}
METADATA: {{"Header_1": "{h1}", "Header_2": "{h2}", "file_url": "{f_url}"}}
"""
        print(search_result)
        return search_result