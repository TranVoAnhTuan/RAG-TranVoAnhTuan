from abc import ABC

from .base import VectorBaseDocument
from .types import DataCategory


class CleanedDocument(VectorBaseDocument, ABC):
    text: str


class CleanPDFTextDocument(CleanedDocument):
    path: str

    class Config:
        name = "cleaned_pdf_text"
        category = DataCategory.PDF_TEXTS
        use_vector_index = False
