from qdrant_client import models
from fastembed import SparseTextEmbedding, TextEmbedding
from sentence_transformers import SentenceTransformer
from llmlingua import PromptCompressor
from FlagEmbedding import FlagReranker
from app.db.qdrant_db import QdrantDatabase
from app.core.config import settings
import asyncio
# from .qwen_reranker import Qwen3Reranker
from app.core.model_manager import model_manager

class RAGService:
    def __init__(self):
        self.db = QdrantDatabase()
        self.client = self.db.get_client()
        self.sparse_model = model_manager.get_sparse_model()
        self.dense_model = model_manager.get_dense_model()

        # self.sparse_model = SparseTextEmbedding(model_name=settings.SPARSE_MODEL_NAME)
        # self.dense_model = SentenceTransformer(settings.DENSE_MODEL_NAME,trust_remote_code=True)
        # self.dense_model = TextEmbedding(model_name="intfloat/multilingual-e5-large")
        
        # self.llm_lingua = PromptCompressor(
        #     model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        #     use_llmlingua2=True,
        # )
        # self.reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, devices="cpu")
        # self.reranker = Qwen3Reranker(use_fp16=True)
        self.reranker = model_manager.get_reranker_model()

    async def retrieve_and_rerank(self, query_text: str) -> str:
        # # 1. Compress prompt
        # compressed_prompt = self.llm_lingua.compress_prompt(
        #     query_text,
        #     rate=0.33,
        #     force_tokens=['\n', '?']
        # )
        # compressed_query = compressed_prompt['compressed_prompt']

        # 1. Embed query
        loop = asyncio.get_event_loop()
        def get_embeddings():
            # Encode trả về numpy array trực tiếp
            dense = self.dense_model.encode(query_text).tolist() 
            sparse = list(self.sparse_model.embed([query_text]))[0]
            return dense, sparse
        # loop = asyncio.get_event_loop()
        # def get_embeddings():
        #     dense = list(self.dense_model.embed([query_text]))[0].tolist()
        #     sparse = list(self.sparse_model.embed([query_text]))[0]
        #     return dense, sparse

        dense_query, sparse_query = await loop.run_in_executor(None, get_embeddings)

        sparse_query_vector = models.SparseVector(
            indices=sparse_query.indices.tolist(),
            values=sparse_query.values.tolist()
        )

        # 2. Hybrid Search
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
            # pairs = [[query_text, doc.payload.get('content', '')] for doc in merged_docs]
            documents = [doc.payload.get('content', '') for doc in merged_docs]
            # return self.reranker.compute_score(pairs)
            return self.reranker.rerank(query_text, documents)

        jina_results = await loop.run_in_executor(None, do_rerank)
        
        # Lấy top 5 kết quả tốt nhất. Jina trả về list of dict có dạng: 
        # {'document': str, 'index': int, 'relevance_score': float}
        top_docs = []
        for result in jina_results[:5]:
            # Dùng 'index' để map ngược lại mảng merged_docs ban đầu
            original_idx = result['index']
            top_docs.append(merged_docs[original_idx])

        search_result = ""
        for i, doc in enumerate(top_docs):
            search_result += f"Result {i+1}: {doc.payload.get('content', '')} citation: (Header 1: {doc.payload.get('Header_1', '')} and Header 2: {doc.payload.get('Header_2', '')} )\n"
            
        return search_result
        # scores = await loop.run_in_executor(None, do_rerank)
        
        # if isinstance(scores, float): scores = [scores]

        # ranked = sorted(zip(merged_docs, scores), key=lambda x: x[1], reverse=True)
        # top_docs = [doc for doc, score in ranked[:5]]

        # search_result = ""
        # for i, doc in enumerate(top_docs):
        #     search_result += f"Result {i+1}: {doc.payload.get('content', '')} citation: (Header 1: {doc.payload.get('Header_1', '')} and Header 2: {doc.payload.get('Header_2', '')} )\n"
            
        # return search_result

        