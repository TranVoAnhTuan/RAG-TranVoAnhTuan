import gc
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes import router, agent


# ── Global Logging Configuration ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────────
    logger.info("🚀 Starting FastAPI server…")

    # Open the MCP connection and build the agent graph.
    # This fetches tools + system prompt from the FastMCP server.
    await agent.connect_mcp()

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────────
    logger.info("🛑 Received shutdown signal. Cleaning up…")

    # Gracefully close the MCP client connection
    await agent.disconnect_mcp()

    # Force Python GC to reclaim any remaining memory
    gc.collect()
    logger.info("🧹 Cleanup complete. Goodbye!")


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