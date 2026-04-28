"""FastAPI route definitions for the RAG backend."""

import logging
import os
import shutil
import traceback

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.agents.demo_agent import DemoAgent
from app.db.sqlite_db import sqlite_db
from pipelines.langgraph_ingestion import ingestion_app

logger = logging.getLogger(__name__)
router = APIRouter()
agent = DemoAgent()


class QueryRequest(BaseModel):
    """Payload for the /chat endpoint."""

    query: str
    thread_id: str = "default_thread"
    topic: str = "General"


class ResumeRequest(BaseModel):
    """Payload for the /chat/resume endpoint."""

    thread_id: str
    decision: str


@router.get("/topics")
async def get_topics() -> dict:
    """Return distinct document topics stored in SQLite."""
    return {"topics": sqlite_db.get_all_topics()}


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...), topic: str = Form("General")) -> dict:
    """Upload a PDF, run the ingestion pipeline, and index into Qdrant."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs("temp_files", exist_ok=True)
    temp_file_path = f"temp_files/{file.filename}"

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        initial_state = {"file_path": temp_file_path, "filename": file.filename, "topic": topic}
        final_state = await ingestion_app.ainvoke(initial_state)

        return {"message": final_state["status"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@router.post("/chat")
async def chat_with_agent(request: QueryRequest) -> dict:
    """Send a query to the agent and return the answer."""
    try:
        answer = await agent.ask(request.query, thread_id=request.thread_id, topic=request.topic)
        return {"answer": answer}
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/history")
async def get_history() -> dict:
    """Return all conversation threads."""
    try:
        return {"threads": await agent.get_all_threads()}
    except Exception as e:
        logger.error(f"History fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/history/{thread_id}")
async def get_thread_messages(thread_id: str) -> dict:
    """Return messages for a specific thread."""
    try:
        return {"messages": await agent.get_thread_messages(thread_id)}
    except Exception as e:
        logger.error(f"Thread messages fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/history/{thread_id}")
async def delete_thread(thread_id: str) -> dict:
    """Delete a conversation thread and its checkpoints."""
    try:
        return {"success": await agent.delete_thread(thread_id)}
    except Exception as e:
        logger.error(f"Thread delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/chat/resume")
async def resume_chat(request: ResumeRequest) -> dict:
    """Resume an interrupted agent execution after HITL approval."""
    try:
        answer = await agent.resume(thread_id=request.thread_id, decision=request.decision)
        return {"answer": answer}
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e)) from e
