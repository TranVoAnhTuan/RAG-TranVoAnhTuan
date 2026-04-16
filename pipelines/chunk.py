import re
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from .state import IngestionState
from app.core.config import settings

def chunk_node(state: IngestionState):
    text = state["cleaned_text"]
    
    has_h1_or_h2 = bool(re.search(r'^(#{1,2})\s', text, re.MULTILINE))
    
    if has_h1_or_h2:
        headers_to_split_on = [("#", "Header_1"), ("##", "Header_2")]
    else:
        headers_to_split_on = [("###", "Header_1")]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(text)

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CONTEXT_LENGTH, 
        chunk_overlap=50,                   
        separators=["\n\n", "\n", ". "]
    )
    
    final_docs = recursive_splitter.split_documents(md_header_splits)

    final_data = []
    for doc in final_docs:
        if not doc.page_content.strip():
            continue
        final_data.append({
            "content": doc.page_content,
            "Header_1": doc.metadata.get("Header_1"),
            "Header_2": doc.metadata.get("Header_2")
        })

    state["chunks"] = final_data
    return state