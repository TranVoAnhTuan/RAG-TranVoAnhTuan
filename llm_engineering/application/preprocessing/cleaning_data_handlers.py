from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from llm_engineering.domain.documents import PDFTextDocument

from llm_engineering.domain.cleaned_documents import (
    CleanPDFTextDocument,
    CleanedDocument,
)

from .operations import clean_text

DocumentT = TypeVar("DocumentT", bound=PDFTextDocument)
CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument)


class CleaningDataHandler(ABC, Generic[DocumentT, CleanedDocumentT]):
    """
    Abstract class for all cleaning data handlers.
    All data transformations logic for the cleaning step is done here
    """

    @abstractmethod
    def clean(self, data_model: DocumentT) -> CleanedDocumentT:
        pass


class PDFCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: PDFTextDocument) -> CleanPDFTextDocument:
        return CleanPDFTextDocument(
            id=data_model.id, path=data_model.path, text=clean_text(data_model.text)
        )
