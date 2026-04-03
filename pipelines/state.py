from typing import List, TypedDict, Optional

class IngestionState(TypedDict):
    file_path: str
    filename: str
    raw_text: str
    cleaned_text: str
    chunks: List[dict]
    tables: List[dict] 
    status: str
    file_hash: Optional[str]
    is_duplicate: Optional[bool]
    minio_url: Optional[str]  # Mới thêm