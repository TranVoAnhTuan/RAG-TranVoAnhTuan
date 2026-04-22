import os
import shutil
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from pipelines.langgraph_ingestion import ingestion_app 
from app.agents.demo_agent import DemoAgent
import traceback
from app.db.sqlite_db import sqlite_db

logger = logging.getLogger(__name__)
router = APIRouter()
agent = DemoAgent()

class QueryRequest(BaseModel):
    query: str
    thread_id: str = "default_thread"
    topic: str = "General"

class ResumeRequest(BaseModel):
    thread_id: str
    decision: str

@router.get("/topics")
async def get_topics():
    return {"topics": sqlite_db.get_all_topics()}

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...), topic: str = Form("General")):
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
            "filename": file.filename,
            "topic": topic
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
        answer = await agent.ask(
            request.query, 
            thread_id=request.thread_id, 
            topic=request.topic
        )
        return {"answer": answer}
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/resume")
async def resume_chat(request: ResumeRequest):
    try:
        answer = await agent.resume(
            thread_id=request.thread_id, 
            decision=request.decision
        )
        return {"answer": answer}
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))