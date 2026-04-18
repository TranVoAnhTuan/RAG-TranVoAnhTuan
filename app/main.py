import gc
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes import router, agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────────
    print("🚀 Starting FastAPI server…")

    # Open the MCP connection and build the agent graph.
    # This fetches tools + system prompt from the FastMCP server.
    await agent.connect_mcp()

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────────
    print("\n🛑 Received shutdown signal. Cleaning up…")

    # Gracefully close the MCP client connection
    await agent.disconnect_mcp()

    # Force Python GC to reclaim any remaining memory
    gc.collect()
    print("🧹 Cleanup complete. Goodbye!\n")


# ── FastAPI application ────────────────────────────────────────────────────────
app = FastAPI(
    title="Agentic RAG PDF API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)