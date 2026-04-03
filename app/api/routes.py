import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from pipelines.langgraph_ingestion import ingestion_app 
from app.agents.demo_agent import DemoAgent

router = APIRouter()
agent = DemoAgent()

class QueryRequest(BaseModel):
    query: str

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    os.makedirs("temp_files", exist_ok=True)
    temp_file_path = f"temp_files/{file.filename}"
    
    # Save to temp_files to avoid loading the entire file into RAM
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        initial_state = {
            "file_path": temp_file_path,
            "filename": file.filename
        }
        
        # Run the ingestion pipeline
        final_state = await ingestion_app.ainvoke(initial_state)
        
        if final_state.get("is_duplicate"):
            return {
                "message": final_state["status"],
                "tables": []
            }
            
        return {
            "message": final_state["status"],
            "tables": final_state.get("tables", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Guarantee cleanup of the temporary file after pipeline finishes
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.post("/chat")
async def chat_with_agent(request: QueryRequest):
    try:
        answer = await agent.ask(request.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))