import re
import uuid
import pymupdf4llm
from zenml import step
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client.models import PointStruct, SparseVector
from qdrant_client import models
from app.db.qdrant_db import QdrantDatabase
from app.core.config import settings
from langchain_text_splitters import MarkdownHeaderTextSplitter

@step
def extract_pdf_step(file_path: str) -> str:
    md_text = pymupdf4llm.to_markdown(file_path)
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

    # Convert the Document objects back into your desired list-of-dictionaries format
    data = []
    for doc in md_header_splits:
        # LangChain stores the extracted headers in the doc.metadata dictionary
        data.append({
            "content": doc.page_content.strip(),
            "Header_1": doc.metadata.get("Header_1"),
            "Header_2": doc.metadata.get("Header_2")
        })

    return data
    # section_pattern = r"^####\s*\d+.*"
    # subsection_pattern = r"^\d+\.\d+\s+.*"

    # data = []
    # current_section = None
    # current_subsection = None
    # current_content = ""

    # blocks = text.split("\n\n")

    # for block in blocks:
    #     block = block.strip()
    #     if not block:
    #         continue

    #     if re.match(section_pattern, block):
    #         if current_section and current_subsection and current_content:
    #             data.append({
    #                 "content": current_content.strip(),
    #                 "Header_1": current_section,
    #                 "Header_2": current_subsection
    #             })
    #         current_section = block
    #         current_subsection = None
    #         current_content = ""

    #     elif re.match(subsection_pattern, block):
    #         if current_section and current_subsection and current_content:
    #             data.append({
    #                 "content": current_content.strip(),
    #                 "Header_1": current_section,
    #                 "Header_2": current_subsection
                    
    #             })
    #         current_subsection = block
    #         current_content = ""

    #     else:
    #         if current_section and current_subsection:
    #             current_content += block + "\n"

    # if current_section and current_subsection and current_content:
    #     data.append({
    #         "content": current_content.strip(),
    #         "Header_1": current_section,
    #         "Header_2": current_subsection
    #     })
    # return data

@step
def embedding_and_load_step(data: list) -> str:
    sparse_model_name = "prithvida/Splade_PP_en_v1"
    dense_model_name = "intfloat/multilingual-e5-large"
    
    sparse_model = SparseTextEmbedding(model_name=sparse_model_name)
    dense_model = TextEmbedding(model_name=dense_model_name)
    
    db = QdrantDatabase()
    client = db.get_client()
    client.create_payload_index(
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
        client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=points
        )
    return f"Successfully processed and loaded {len(points)} chunks into Qdrant."