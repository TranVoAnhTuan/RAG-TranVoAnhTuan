from langgraph.graph import StateGraph, START, END
from .state import IngestionState
from .extract import extract_node
from .clean import clean_node
from .chunk import chunk_node
from .embed import embed_and_load_node

def build_ingestion_graph():
    workflow = StateGraph(IngestionState)

    workflow.add_node("extract", extract_node)
    workflow.add_node("clean", clean_node)
    workflow.add_node("chunk", chunk_node)
    workflow.add_node("embed_and_load", embed_and_load_node)

    workflow.add_edge(START, "extract")
    workflow.add_edge("extract", "clean")
    workflow.add_edge("clean", "chunk")
    workflow.add_edge("chunk", "embed_and_load")
    workflow.add_edge("embed_and_load", END)

    return workflow.compile()

ingestion_app = build_ingestion_graph()


# import os
# import gc
# import re
# import uuid
# import asyncio
# from typing import List, TypedDict
# from langgraph.graph import StateGraph, START, END
# import pandas as pd

# # Import Docling
# from docling.document_converter import DocumentConverter, PdfFormatOption
# from docling.datamodel.base_models import InputFormat
# from docling.datamodel.pipeline_options import PdfPipelineOptions
# from docling.datamodel.accelerator_options import AcceleratorOptions

# from fastembed import TextEmbedding, SparseTextEmbedding
# from qdrant_client.models import PointStruct, SparseVector
# from qdrant_client import models
# from app.db.qdrant_db import QdrantDatabase
# from app.core.config import settings
# from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
# tokenizer = model.tokenizer



# class IngestionState(TypedDict):
#     file_path: str
#     raw_text: str
#     cleaned_text: str
#     chunks: List[dict]
#     tables: List[dict] 
#     status: str

# def _process_with_docling(file_path: str) -> dict:
#     print("Loading Docling...")

#     pipeline_options = PdfPipelineOptions()
#     pipeline_options.do_ocr = True
#     pipeline_options.do_table_structure = True

#     pipeline_options.accelerator_options = AcceleratorOptions(
#         num_threads=6,
#         device="cpu"
#     )

#     converter = DocumentConverter(
#         format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
#     )

#     result = converter.convert(file_path)
#     doc = result.document

#     tables_data = []
#     if hasattr(doc, 'tables'):
#         for i, table in enumerate(doc.tables, 1):
#             try:
#                 df = table.export_to_dataframe()

#                 df = df.replace(r'^[-|\s]*$', pd.NA, regex=True)
#                 df = df.dropna(how='all', axis=0)
#                 df = df.fillna("")
#                 if df.empty:
#                     continue

#                 tables_data.append({
#                     "table_number": i,
#                     "data": df.to_dict(orient="records")
#                 })
#             except Exception as e:
#                 print(f"Table error {i}: {e}")

#     markdown_text = doc.export_to_markdown()

#     print("Cleaning RAM...")
#     del converter, result, doc
#     gc.collect()

#     return {
#         "markdown": markdown_text,
#         "tables": tables_data
#     }


# async def extract_node(state: IngestionState):
#     file_path = state["file_path"]
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"No file found at: {file_path}")
    
#     docling_result = await asyncio.to_thread(_process_with_docling, file_path)
    
#     return {
#         "raw_text": docling_result["markdown"],
#         "tables": docling_result["tables"] 
#     }

# def clean_node(state: IngestionState):
#     text = state["raw_text"]
#     text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
#     text = re.sub(r'-{4,}', '--', text)
#     text = re.sub(r'^## (?!\d+\.\d+\b)(.*)', r'### \1', text, flags=re.MULTILINE)
#     text = re.sub(r'(?m)^### (.+)\n\s*### (\d+\.(?:\d+)?\s+.*)', r'# \1 - \2',text)    
#     text = re.sub(r'(?m)^### (\d+\.\d+)\s+(.*)', r'## \1 \2', text)

#     text = re.sub(r'\n\n(?=- )', '\n', text)
#     text = re.sub(r'\n\n\s+(?=- )', ' ', text)
#     text = re.sub(r'(?m)^(\s*-\s+)\S+\s+', r'\1', text)
#     text = re.sub(r'[ \t]{2,}', ' ', text)
#     text = re.sub(r'\n{3,}', '\n\n', text)
    
#     return {"cleaned_text": text}

# def split_by_token_with_paragraph(text, tokenizer, max_tokens=512):
#     tokens = tokenizer.encode(text, add_special_tokens=False)
#     chunks = []

#     start = 0
#     while start < len(tokens):
#         end = start + max_tokens
#         chunk_tokens = tokens[start:end]

#         chunk_text = tokenizer.decode(chunk_tokens)

#         # ưu tiên cắt tại paragraph
#         split_pos = chunk_text.rfind("\n\n")

#         if split_pos != -1 and end < len(tokens):
#             chunk_text = chunk_text[:split_pos]
#             actual_tokens = tokenizer.encode(chunk_text, add_special_tokens=False)
#             end = start + len(actual_tokens)

#         chunks.append(chunk_text.strip())
#         start = end

#     return chunks

# def chunk_node(state: dict):
#     text = state["cleaned_text"]

#     headers_to_split_on = [
#         ("#", "Header_1"),
#         ("##", "Header_2"),
#     ]

#     markdown_splitter = MarkdownHeaderTextSplitter(
#         headers_to_split_on=headers_to_split_on,
#         strip_headers=False
#     )

#     md_header_splits = markdown_splitter.split_text(text)

#     final_data = []

#     for doc in md_header_splits:
#         sub_chunks = split_by_token_with_paragraph(
#             doc.page_content,
#             tokenizer=tokenizer,
#             max_tokens=512   
#         )

#         for sub_content in sub_chunks:
#             if not sub_content:
#                 continue

#             final_data.append({
#                 "content": sub_content,
#                 "Header_1": doc.metadata.get("Header_1"),
#                 "Header_2": doc.metadata.get("Header_2"),
#             })

#     return {"chunks": final_data}

# async def embed_and_load_node(state: IngestionState):
#     chunks = state["chunks"]
#     sparse_model = SparseTextEmbedding(model_name=settings.SPARSE_MODEL_NAME)
#     dense_model = TextEmbedding(model_name=settings.DENSE_MODEL_NAME)
    
#     db = QdrantDatabase()
#     client = db.get_client()
#     await db._ensure_collection_exists()

#     await client.create_payload_index(
#         collection_name=settings.QDRANT_COLLECTION_NAME,
#         field_name="Header_1",
#         field_schema=models.PayloadSchemaType.KEYWORD
#     )

#     points = []
#     for item in chunks:
#         text = item["content"]
#         # Skip empty chunks
#         if not text:
#             continue
            
#         dense_vec = list(dense_model.embed([text]))[0].tolist()
#         sparse_vec = list(sparse_model.embed([text]))[0]

#         point = PointStruct(
#             id=str(uuid.uuid4()),
#             vector={
#                 "dense": dense_vec,
#                 "sparse": SparseVector(indices=sparse_vec.indices.tolist(), values=sparse_vec.values.tolist())
#             },
#             payload={
#                 "content": item["content"],
#                 "Header_1": item["Header_1"],
#                 "Header_2": item["Header_2"]
#                 }
#         )
#         points.append(point)

#     if points:
#         batch_size = 10  
#         for i in range(0, len(points), batch_size):
#             batch = points[i : i + batch_size]
#             await client.upsert(collection_name=settings.QDRANT_COLLECTION_NAME, points=batch)
#             await asyncio.sleep(0.5) 

#     return {"status": f"The document was processed, and {len(points)} chunks were saved to Qdrant successfully!"}

# workflow = StateGraph(IngestionState)

# workflow.add_node("extract", extract_node)
# workflow.add_node("clean", clean_node)
# workflow.add_node("chunk", chunk_node)
# workflow.add_node("embed_and_load", embed_and_load_node)

# workflow.add_edge(START, "extract")
# workflow.add_edge("extract", "clean")
# workflow.add_edge("clean", "chunk")
# workflow.add_edge("chunk", "embed_and_load")
# workflow.add_edge("embed_and_load", END)

# ingestion_app = workflow.compile()