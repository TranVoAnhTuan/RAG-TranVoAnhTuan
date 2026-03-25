from typing import List, TypedDict

class IngestionState(TypedDict):
    file_path: str
    raw_text: str
    cleaned_text: str
    chunks: List[dict]
    tables: List[dict] 
    status: str