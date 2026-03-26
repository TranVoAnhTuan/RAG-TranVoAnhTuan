import gc
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI): 
    print("🚀 Start the FastAPI server...")
    yield 

    print("\n🛑 Received command to stop server (Ctrl+C). Resource cleanup in progress...")

    # Remove FastEmbed models in RAGService
    try:
        from app.api.routes import agent
        from app.agents.demo_agent import rag_service
        
        del rag_service.sparse_model
        
        # Delete the entire instance
        del rag_service
        del agent
        print("✅ Freed Embedding and Reranker models.")
    except Exception as e:
        pass

    # Force Python to clean memory immediately
    gc.collect()
    print("🧹 RAM has been completely freed. Goodbye!\n")

# Initialize FastAPI with lifespan
app = FastAPI(
    title="Agentic RAG PDF API", 
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    # Note: When using reload=True, uvicorn might spawn multiple workers.
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)