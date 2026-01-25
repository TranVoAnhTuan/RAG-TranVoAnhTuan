from abc import ABC

from pydantic import Field

from llm_engineering.domain.base import VectorBaseDocument
from llm_engineering.domain.types import DataCategory


class Chunk(VectorBaseDocument, ABC):
    text: str
    metadata: dict = Field(default_factory=dict)


class PDFTextChunk(Chunk):
    path: str

    class Config:
        category = DataCategory.PDF_TEXTS
