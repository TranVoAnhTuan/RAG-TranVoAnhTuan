from qdrant_client import models
from fastembed import SparseTextEmbedding, TextEmbedding
from llmlingua import PromptCompressor
from FlagEmbedding import FlagReranker
from app.db.qdrant_db import QdrantDatabase
from app.core.config import settings

class RAGService:
    def __init__(self):
        self.db = QdrantDatabase()
        self.client = self.db.get_client()
        self.sparse_model = SparseTextEmbedding(model_name="prithvida/Splade_PP_en_v1")
        self.dense_model = TextEmbedding(model_name="intfloat/multilingual-e5-large")
        
        # self.llm_lingua = PromptCompressor(
        #     model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        #     use_llmlingua2=True,
        # )
        self.reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

    def retrieve_and_rerank(self, query_text: str) -> str:
        # # 1. Compress prompt
        # compressed_prompt = self.llm_lingua.compress_prompt(
        #     query_text,
        #     rate=0.33,
        #     force_tokens=['\n', '?']
        # )
        # compressed_query = compressed_prompt['compressed_prompt']

        # 2. Embed query
        dense_query = list(self.dense_model.embed([query_text]))[0].tolist()
        sparse_query = list(self.sparse_model.embed([query_text]))[0]
        
        sparse_query_vector = models.SparseVector(
            indices=sparse_query.indices.tolist(),
            values=sparse_query.values.tolist()
        )

        # 3. Hybrid Search
        hybrid_results = self.client.query_points(
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
            return "Không tìm thấy thông tin phù hợp trong tài liệu."

        merged_docs_dict = {doc.id: doc for doc in hybrid_results.points}

        processed_headers = set()
        for doc in hybrid_results.points:
            header = doc.payload.get("Header_1")
            if header and header not in processed_headers:
                processed_headers.add(header)
                try:
                    filter_points, _ = self.client.scroll(
                        collection_name = settings.QDRANT_COLLECTION_NAME,
                        scroll_filters = models.Filter(
                            must = [models.FieldCondition(key= "Header_1", match = models.MatchValue(value=header))]
                        ),
                        limit = 10,
                        with_payload = True,
                    )
                    for p in filter_points:
                        merged_docs_dict[p.id] = p
                except Exception as e:
                    pass

        merged_docs = list(merged_docs_dict.values())

        if not merged_docs:
            return "Do not find the apropriate document"

        # 4. Reranking
        pairs = [[query_text, doc.payload['content']] for doc in merged_docs]
        scores = self.reranker.compute_score(pairs)
        
        if isinstance(scores, float):
            scores = [scores]

        ranked = sorted(zip(merged_docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in ranked[:5]]

        # 6. Format results
        search_result = ""
        for i, doc in enumerate(top_docs):
            search_result += f"Result {i+1}: {doc.payload.get('content', '')} \n"
            
        return search_result

        