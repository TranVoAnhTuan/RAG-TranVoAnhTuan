import re
import uuid
import pymupdf4llm
from zenml import step
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client.models import PointStruct, SparseVector
from qdrant_client import models
from app.db.qdrant_db import QdrantDatabase
from app.core.config import settings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import os
import asyncio
@step
def extract_pdf_step(file_path: str) -> str:
    md_text = pymupdf4llm.to_markdown(file_path, margins=(50, 50, 50, 50))
    return md_text



@step
def cleaning_step(text: str) -> str:
    text = re.split(r'(?mi)^##\s*contact\s*us', text)[0]
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\*\*\d+\*\*\s*\*\*/\*\*\s*\w+', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\n\n(?=- )', '\n', text)
    text = re.sub(r'\n\n\s+(?=- )', ' ', text)  
    text = re.sub(r'\n\n(?=[a-z])', ' ', text)
    text = re.sub(r'(?m)^(\d+)\s+(?!\d+\.)', r'#### \1 ', text)
    text = re.sub(r'(?m)^\*\*(\d+.*?)\*\*', r'#### \1', text)
    text = re.sub(r'(?m)^(\d+\.\d+)\s+(.*)', r'##### \1 \2', text)
    text = re.sub(r'#\s*Contents[\s\S]*?(?=#+\s*\d)', '', text, flags=re.IGNORECASE)  
    return text

@step
def chunking_step(text: str) -> list:
    # Define the Markdown headers you want to split on and their metadata keys
    headers_to_split_on = [
        ("####", "Header_1"),
        ("#####", "Header_2") # Assuming your subsections use H5
    ]

    # Initialize the splitter
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False # Keeps the header text in the content (set True to remove)
    )

    # Split the text into LangChain Document objects
    md_header_splits = markdown_splitter.split_text(text)

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, 
        chunk_overlap=50, 
        length_function=len 
    )

    final_data = []
    for doc in md_header_splits:
        if len(doc.page_content) > 1024:
            sub_chunks = child_splitter.split_text(doc.page_content)
            for sub_content in sub_chunks:
                final_data.append({
                    "content": sub_content.strip(),
                    "Header_1": doc.metadata.get("Header_1"),
                    "Header_2": doc.metadata.get("Header_2")
                })
        else:
            final_data.append({
                "content": doc.page_content.strip(),
                "Header_1": doc.metadata.get("Header_1"),
                "Header_2": doc.metadata.get("Header_2")
            })

    return final_data

@step
def embedding_and_load_step(data: list) -> str:
    # 1. Khai báo một hàm async nội bộ chứa toàn bộ logic tương tác với database
    async def _async_process():
        sparse_model = SparseTextEmbedding(model_name=settings.SPARSE_MODEL_NAME)
        dense_model = TextEmbedding(model_name=settings.DENSE_MODEL_NAME)
        
        db = QdrantDatabase()
        client = db.get_client() 

        await db._ensure_collection_exists()

        await client.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            field_name="Header_1",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        
        points = []
        for item in data:
            text = item["content"]
            dense_vec = list(dense_model.embed([text]))[0].tolist()
            sparse_vec = list(sparse_model.embed([text]))[0]

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vec,
                    "sparse": SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist() 
                    )
                },
                payload={
                    "content": item["content"],
                    "Header_1": item["Header_1"],
                    "Header_2": item["Header_2"]
                }
            )
            points.append(point)

        if points:
            batch_size = 10  
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                await client.upsert(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    points=batch
                )
                print(f"Đã upsert batch {i // batch_size + 1} ({len(batch)} points)")
        return f"Successfully processed and loaded {len(points)} chunks into Qdrant."

    return asyncio.run(_async_process())