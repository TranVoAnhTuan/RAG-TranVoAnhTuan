from pydantic import BaseModel, Field
from typing import List, Optional

class Citation(BaseModel):
    Header_1: Optional[str] = Field(default="", description="Header 1 of the source document.")
    Header_2: Optional[str] = Field(default="", description="Header 2 of the source document.")
    file_url: Optional[str] = Field(default="", description="The URL path to the source file in MinIO.")

class DocumentInfo(BaseModel):
    response: str = Field(
        description="A very concise and direct answer to the user query."
    )
    citations: List[Citation] = Field(
        description="List of relevant sources used for the response."
    )