from typing import List

from loguru import logger

from llm_engineering.application.networks import EmbeddingModelSingleton

embedding_model = EmbeddingModelSingleton()


def chunk_text(
    text: str,
    max_tokens: int = 500,
    overlap: int = 50,
) -> List[str]:
    """
    Split input text into overlapping chunks based on tokenizer token length.

    Args:
        text (str): Input text to chunk.
        max_tokens (int): Maximum tokens per chunk.
        overlap (int): Number of overlapping tokens between chunks.

    Returns:
        List[str]: List of text chunks.
    """

    if not text.strip():
        return []

    tokenizer = embedding_model.tokenizer

    sentences = text.split(". ")
    token_lengths = [
        len(tokenizer.encode(sentence, add_special_tokens=False))
        for sentence in sentences
    ]

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_tokens = 0

    for sentence, token_len in zip(sentences, token_lengths):
        # If adding this sentence would exceed the limit
        if current_tokens + token_len > max_tokens:
            if current_chunk:
                chunk_text = ". ".join(current_chunk).strip() + "."
                chunks.append(chunk_text)

            # Build overlap from previous chunk
            if chunks and overlap > 0:
                overlap_tokens = tokenizer.encode(chunks[-1], add_special_tokens=False)[
                    -overlap:
                ]
                overlap_text = tokenizer.decode(overlap_tokens).strip()
                current_chunk = [overlap_text]
                current_tokens = len(overlap_tokens)
            else:
                current_chunk = []
                current_tokens = 0

        current_chunk.append(sentence)
        current_tokens += token_len

    if current_chunk:
        chunks.append(". ".join(current_chunk).strip() + ".")

    logger.debug(f"Chunked text into {len(chunks)} chunks")

    return chunks
