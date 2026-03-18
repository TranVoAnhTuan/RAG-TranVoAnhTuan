import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from pipelines.ingestion import process_pdf_pipeline
from app.agents.demo_agent import DemoAgent

router = APIRouter()
agent = DemoAgent()

class QueryRequest(BaseModel):
    query: str

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file PDF.")
    
    # Lưu file tạm thời để ZenML xử lý
    os.makedirs("temp_files", exist_ok=True)
    temp_file_path = f"temp_files/{file.filename}"
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Chạy ZenML Pipeline
        process_pdf_pipeline(temp_file_path)
        return {"message": "Tài liệu đã được xử lý và lưu vào Qdrant thành công!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Dọn dẹp file tạm
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.post("/chat")
async def chat_with_agent(request: QueryRequest):
    try:
        answer = agent.ask(request.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))