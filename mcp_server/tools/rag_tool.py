from typing import Optional

from fastmcp import FastMCP

from mcp_server.rag_service import rag_service


def register_tools(mcp: FastMCP) -> None:
    """
    Register all RAG tools onto the FastMCP instance.

    Each tool is decorated with @mcp.tool() so it is automatically advertised
    to MCP clients (e.g. the FastAPI backend's DemoAgent) via the tool-listing
    endpoint.
    """

    @mcp.tool(
        name="search_document_knowledge",
        description=(
            "Search for information in the document knowledge base. "
            "You MUST provide the `topic` parameter if you are confident the "
            "user's query belongs to a specific topic. "
            "If the topic is unknown or unclear, do not provide this parameter. "
            "IMPORTANT GUIDELINE: If called a 2nd or 3rd time, the query must "
            "represent a completely different approach or a broader topic."
        ),
    )
    async def search_document_knowledge(
        query: str, topic: Optional[str] = None
    ) -> str:
        return await rag_service.retrieve_and_rerank(query, filter_topic=topic)
