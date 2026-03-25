import gc
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ==========================================
    # 1. Khởi chạy (Startup)
    # ==========================================
    print("🚀 Khởi động server FastAPI...")
    yield  # Server đang chạy ở đây...
    
    # ==========================================
    # 2. Tắt server (Shutdown) - Giải phóng RAM
    # ==========================================
    print("\n🛑 Đã nhận lệnh dừng server (Ctrl+C). Đang dọn dẹp tài nguyên...")

    # Xóa mô hình Docling
    try:
        from pipelines.langgraph_ingestion import GLOBAL_CONVERTER
        del GLOBAL_CONVERTER
        print("✅ Đã giải phóng Docling Converter.")
    except Exception as e:
        pass

    # Xóa các mô hình FastEmbed và Reranker trong RAGService
    try:
        from app.api.routes import agent
        from app.agents.demo_agent import rag_service
        
        # Xóa các mô hình nặng bên trong class
        del rag_service.dense_model
        del rag_service.sparse_model
        del rag_service.reranker
        
        # Xóa toàn bộ instance
        del rag_service
        del agent
        print("✅ Đã giải phóng Embedding và Reranker models.")
    except Exception as e:
        pass

    # Ép Python dọn dẹp bộ nhớ ngay lập tức
    gc.collect()
    print("🧹 RAM đã được giải phóng hoàn toàn. Tạm biệt!\n")

# Khởi tạo FastAPI với lifespan
app = FastAPI(
    title="Agentic RAG PDF API", 
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    # Lưu ý: Khi dùng reload=True, uvicorn có thể sinh ra nhiều worker.
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)