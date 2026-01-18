from abc import ABC
from typing import Optional

from pydantic import UUID4, Field

from .base import NoSQLBaseDocument
from .types import DataCategory

class PDFTextDocument(NoSQLBaseDocument):
    path: str = Field(description="Absolute path to the PDF file")
    text: str

    class Settings:
        name = DataCategory.PDF_TEXTS


