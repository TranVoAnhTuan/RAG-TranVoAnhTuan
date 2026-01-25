from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application import utils
from llm_engineering.application.preprocessing import (
    ChunkingDispatcher,
    EmbeddingDispatcher,
)
from llm_engineering.domain.chunks import Chunk
from llm_engineering.domain.embedded_chunks import EmbeddedChunk


@step
def chunk_and_embed(
    cleaned_documents: Annotated[list, "cleaned_documents"],
) -> Annotated[list, "embedded_documents"]:
    metadata = {
        "chunking": {},
        "embedding": {},
        "num_documents": len(cleaned_documents),
    }

    embedded_chunks = []
    for document in cleaned_documents:
        chunks = ChunkingDispatcher.dispatch(document)
        metadata["chunking"] = _add_chunks_metadata(chunks, metadata["chunking"])

        for batched_chunks in utils.misc.batch(chunks, 10):
            batched_embedded_chunks = EmbeddingDispatcher.dispatch(batched_chunks)
            embedded_chunks.extend(batched_embedded_chunks)

    metadata["embedding"] = _add_embeddings_metadata(
        embedded_chunks, metadata["embedding"]
    )
    metadata["num_chunks"] = len(embedded_chunks)
    metadata["num_embedded_chunks"] = len(embedded_chunks)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="embedded_documents", metadata=metadata
    )

    return embedded_chunks


def _add_chunks_metadata(chunks: list[Chunk], metadata: dict) -> dict:
    if not chunks:
        return metadata

    category = chunks[0].get_category()

    if category not in metadata:
        metadata[category] = {
            "num_chunks": 0,
            "sources": set(),
        }
    sources = set(metadata[category].get("sources", []))

    for chunk in chunks:
        metadata[category]["num_chunks"] += 1
        sources.add(chunk.path)

    metadata[category]["sources"] = list(sources)

    return metadata


def _add_embeddings_metadata(
    embedded_chunks: list[EmbeddedChunk], metadata: dict
) -> dict:
    if not embedded_chunks:
        return metadata

    category = embedded_chunks[0].get_category()

    if category not in metadata:
        metadata[category] = {
            "sources": set(),
        }

    for embedded_chunk in embedded_chunks:
        metadata[category]["sources"].add(embedded_chunk.path)

    metadata[category]["sources"] = list(metadata[category]["sources"])

    return metadata
