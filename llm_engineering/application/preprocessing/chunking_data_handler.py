import hashlib
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import UUID

from llm_engineering.domain.chunks import Chunk, PDFTextChunk
from llm_engineering.domain.cleaned_documents import CleanPDFTextDocument

from .operations import chunk_text

CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanPDFTextDocument)
ChunkT = TypeVar("ChunkT", bound=Chunk)


class ChunkingDataHandler(ABC, Generic[CleanedDocumentT, ChunkT]):
    """
    Abstract class for all Chunking data handlers.
    All data transformations logic for the chunking step is done here
    """

    @property
    def metadata(self) -> dict:
        return {
            "chunk_size": 500,
            "chunk_overlap": 50,
        }

    @abstractmethod
    def chunk(self, data_model: CleanedDocumentT) -> list[ChunkT]:
        pass


class PDFTextChunkingHandler(ChunkingDataHandler[CleanPDFTextDocument, PDFTextChunk]):
    def chunk(self, data_model: CleanPDFTextDocument) -> list[PDFTextChunk]:
        chunks_data: list[PDFTextChunk] = []

        chunks = chunk_text(
            data_model.text,
            max_tokens=self.metadata["chunk_size"],
            overlap=self.metadata["chunk_overlap"],
        )

        for chunk in chunks:
            # Stable deterministic UUID from content
            chunk_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()
            chunk_id = UUID(chunk_hash, version=4)

            model = PDFTextChunk(
                id=chunk_id,
                path=data_model.path,
                text=chunk,
                metadata=self.metadata,
            )

            chunks_data.append(model)

        return chunks_data
