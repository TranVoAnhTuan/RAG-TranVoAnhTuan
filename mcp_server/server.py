"""
FastMCP server entry point for the RAG Tools Service.

Exposes:
  • Tool:   search_document_knowledge  — hybrid RAG retrieval + rerank
  • Prompt: rag_system_prompt          — system prompt for the agent

Transport: Streamable HTTP (default FastMCP v2 transport)
  Clients connect to:  http://<host>:8001/mcp
"""

import logging

from fastmcp import FastMCP

from mcp_server.prompts.system_prompt import register_prompt
from mcp_server.tools.final_answer_tool import register_tools as register_final_answer_tool
from mcp_server.tools.rag_tool import register_tools
from mcp_server.tools.tavily_tool import register_tavily_tool

# ── Global Logging Configuration ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _ensure_models() -> None:
    """Pre-download Dense, Reranker, and Sparse models if not cached."""
    from huggingface_hub import snapshot_download

    from mcp_server.config import mcp_settings

    for name in (mcp_settings.DENSE_MODEL_NAME, mcp_settings.RERANKER_MODEL_NAME):
        logger.info(f"📦 Checking model: {name} …")
        snapshot_download(name)
        logger.info(f"✅ Model ready: {name}")

    logger.info(f"📦 Checking Sparse model: {mcp_settings.SPARSE_MODEL_NAME} …")
    from fastembed import SparseTextEmbedding

    SparseTextEmbedding(model_name=mcp_settings.SPARSE_MODEL_NAME)
    logger.info(f"✅ Sparse model ready: {mcp_settings.SPARSE_MODEL_NAME}")


# ── Pre-download models at import time ─────────────────────────────────────────
_ensure_models()

# ── Create FastMCP application ─────────────────────────────────────────────────
mcp = FastMCP(
    name="RAG Tools Server",
    instructions=(
        "This server provides document retrieval tools and the agent system prompt "
        "for the Agentic RAG system. Use `search_document_knowledge` to retrieve "
        "relevant passages from the knowledge base."
    ),
)

# ── Register tools and prompts ─────────────────────────────────────────────────
register_tools(mcp)
register_tavily_tool(mcp)
register_final_answer_tool(mcp)
register_prompt(mcp)


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    # Expose the FastMCP ASGI app via uvicorn.
    # The MCP endpoint is available at:  POST /mcp  (Streamable HTTP transport)
    app = mcp.http_app(path="/mcp")
    uvicorn.run(app, host="0.0.0.0", port=8001)
