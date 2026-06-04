from typing import TypedDict


class IngestionState(TypedDict):
    file_path: str
    filename: str
    raw_text: str
    cleaned_text: str
    chunks: list[dict]
    status: str
    file_hash: str | None
    is_duplicate: bool | None
    minio_url: str | None
    topic: str | None
    title: str
