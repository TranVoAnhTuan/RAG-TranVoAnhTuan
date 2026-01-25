from abc import ABC

from pydantic import Field

from .base import NoSQLBaseDocument
from .types import DataCategory


class Document(NoSQLBaseDocument, ABC):
    text: str


class PDFTextDocument(Document):
    path: str = Field(description="Absolute path to the PDF file")

    class Settings:
        name = DataCategory.PDF_TEXTS
