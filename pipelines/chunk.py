import gc
import logging

from langchain_text_splitters import MarkdownHeaderTextSplitter

from app.core.config import settings
from app.core.model_manager import model_manager

from .state import IngestionState

logger = logging.getLogger(__name__)


def split_by_token_with_paragraph(text: str, tokenizer: object, max_tokens: int = 512) -> list[str]:
    """Split text into token-bounded chunks, preferring paragraph boundaries."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        split_pos = chunk_text.rfind("\n\n")

        if split_pos != -1 and end < len(tokens):
            chunk_text = chunk_text[:split_pos]
            actual_tokens = tokenizer.encode(chunk_text, add_special_tokens=False)
            end = start + len(actual_tokens)

        chunks.append(chunk_text.strip())
        start = end
    return chunks


def chunk_node(state: IngestionState) -> dict:
    logger.info("Loading Tokenizer for chunking...")
    # Load tokenizer locally
    tokenizer = model_manager.get_tokenizer()

    text = state["cleaned_text"]

    # ── Build document-title prefix ─────────────────────────────────
    title = state.get("title", "")
    title_prefix = ""
    if title:
        headings = [h.strip() for h in title.split(" | ")]
        title_prefix = f"title: {headings}\n\n"

    headers_to_split_on = [("#", "Header_1"), ("##", "Header_2")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    md_header_splits = markdown_splitter.split_text(text)

    # If any section has no header metadata, the document probably uses ###
    # headings (e.g. after clean.py conversion). Re-split those sections with ###.
    if any(not d.metadata for d in md_header_splits):
        h3_splitter = MarkdownHeaderTextSplitter([("###", "Header_1")], strip_headers=False)
        resolved: list = []
        for d in md_header_splits:
            if d.metadata:
                resolved.append(d)
            else:
                sub = h3_splitter.split_text(d.page_content)
                resolved.extend(sub)
        md_header_splits = resolved

    final_data = []
    for doc in md_header_splits:
        sub_chunks = split_by_token_with_paragraph(
            doc.page_content, tokenizer=tokenizer, max_tokens=settings.CONTEXT_LENGTH
        )
        for sub_content in sub_chunks:
            if not sub_content:
                continue
            enriched_content = title_prefix + sub_content if title_prefix else sub_content
            final_data.append(
                {
                    "content": enriched_content,
                    "Header_1": doc.metadata.get("Header_1"),
                    "Header_2": doc.metadata.get("Header_2"),
                }
            )

    logger.info("Cleaning Tokenizer RAM...")
    del tokenizer
    gc.collect()

    return {"chunks": final_data}
