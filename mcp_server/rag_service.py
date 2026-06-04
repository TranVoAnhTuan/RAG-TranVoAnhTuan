import asyncio
import logging

from qdrant_client import AsyncQdrantClient, models

from mcp_server.config import mcp_settings
from mcp_server.model_manager import model_manager

logger = logging.getLogger(__name__)


class RAGService:
    """
    Handles hybrid retrieval (Dense + Sparse via Qdrant RRF fusion) and
    cross-encoder reranking for the MCP server.

    This is a self-contained copy of app/services/rag_service.py with all
    imports pointing to mcp_server.* so the MCP container has no dependency
    on the FastAPI app package.
    """

    def __init__(self) -> None:
        self.client = AsyncQdrantClient(
            url=mcp_settings.QDRANT_URL,
            api_key=mcp_settings.QDRANT_API_KEY,
            timeout=120,
        )
        self.collection_name: str = mcp_settings.QDRANT_COLLECTION_NAME
        # Sparse model lives on CPU permanently
        self.sparse_model = model_manager.get_sparse_model()

    async def retrieve_and_rerank(self, query_text: str, filter_topic: str | None = None) -> str:
        """Full retrieval pipeline: embed → hybrid search → expand context → rerank → format."""
        loop = asyncio.get_event_loop()

        # ── 1. Embed ──────────────────────────────────────────────────
        def get_embeddings():
            with model_manager.use_dense_model_on_gpu() as dense_model:
                dense = dense_model.encode(query_text).tolist()
            sparse = next(iter(self.sparse_model.embed([query_text])))
            return dense, sparse

        dense_query, sparse_query = await loop.run_in_executor(None, get_embeddings)

        sparse_query_vector = models.SparseVector(
            indices=list(sparse_query.indices) if sparse_query.indices is not None else [],
            values=list(sparse_query.values) if sparse_query.values is not None else [],
        )

        # ── 2. Build optional topic filter ───────────────────────────
        query_filter = None
        if filter_topic:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="topic",
                        match=models.MatchValue(value=filter_topic),
                    )
                ]
            )

        # ── 3. Hybrid Search (RRF) ────────────────────────────────────
        hybrid_results = await self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_query,
                    using="dense",
                    limit=20,
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=sparse_query_vector,
                    using="sparse",
                    limit=20,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=10,
        )

        if not hybrid_results.points:
            return "No appropriate information was found."

        # ── 4. Context-window expansion ───────────────────────────────
        merged_docs_dict = {doc.id: doc for doc in hybrid_results.points}

        # Collect unique (title, Header_1) pairs — scopes expansion to
        # the same document *and* section, preventing cross-document
        # collisions when two documents share identical Header_1 values.
        processed_sections = {
            (doc.payload.get("title") or "", doc.payload.get("Header_1"))
            for doc in hybrid_results.points
            if doc.payload.get("Header_1")
        }

        for doc_title, header_val in processed_sections:
            try:
                conditions = [
                    models.FieldCondition(
                        key="Header_1",
                        match=models.MatchValue(value=header_val),
                    ),
                ]
                # Old points may lack the "title" field — fall back to
                # Header_1-only expansion for backward compatibility.
                if doc_title:
                    conditions.insert(
                        0,
                        models.FieldCondition(
                            key="title",
                            match=models.MatchValue(value=doc_title),
                        ),
                    )

                filter_points, _ = await self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=models.Filter(must=conditions),
                    limit=10,
                    with_payload=True,
                )
                for p in filter_points:
                    merged_docs_dict[p.id] = p
            except Exception:
                continue

        merged_docs = list(merged_docs_dict.values())

        # ── 5. Rerank ─────────────────────────────────────────────────
        def do_rerank():
            documents = [doc.payload.get("content", "") for doc in merged_docs]
            with model_manager.use_reranker_model() as reranker:
                scores = reranker.rerank(query_text, documents)
            return scores

        jina_results = await loop.run_in_executor(None, do_rerank)

        top_docs = [merged_docs[r["index"]] for r in jina_results[:5]]

        # ── 6. Format output ──────────────────────────────────────────
        search_result = "SEARCH RESULTS:\n"
        for i, doc in enumerate(top_docs):
            h1 = doc.payload.get("Header_1", "")
            h2 = doc.payload.get("Header_2", "")
            f_url = doc.payload.get("file_url", "")
            content = doc.payload.get("content", "")
            search_result += f"""
--- Result {i + 1} ---
CONTENT: {content}
METADATA: {{"Header_1": "{h1}", "Header_2": "{h2}", "file_url": "{f_url}"}}
"""
        # NOTE: Old data (indexed before MINIO_PUBLIC_ENDPOINT existed) stores
        # presigned URLs with the internal hostname "minio:9000". Those links
        # are broken from the browser because the hostname doesn't resolve, and
        # rewriting them here would also break the MinIO cryptographic signature.
        # To fix old citations, re-upload and re-index the source PDFs.

        logger.info(search_result)
        return search_result


# Singleton — shared across the MCP server process
rag_service = RAGService()
