from abc import ABC

from pydantic import Field

from llm_engineering.domain.types import DataCategory
from .base import VectorBaseDocument


class EmbeddedChunk(VectorBaseDocument, ABC):
    text: str
    embedding: list[float] | None = None
    path: str
    metadata: dict = Field(default_factory=dict)

    @classmethod
    def to_context(cls, chunks: list["EmbeddedChunk"]) -> str:
        context = ""
        for i, chunk in enumerate(chunks):
            context += f"""
            Chunk {i + 1}:
            Type: {chunk.__class__.__name__}
            Source: {chunk.path}
            Content: {chunk.text}\n
            """
        return context


class EmbeddedPDFTextChunk(EmbeddedChunk):
    class Config:
        name = "embedded_pdf_chunks"
        category = DataCategory.PDF_TEXTS
        use_vector_index = True
