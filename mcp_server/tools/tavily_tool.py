from fastmcp import FastMCP
from tavily import TavilyClient
from mcp_server.config import mcp_settings

def register_tavily_tool(mcp: FastMCP) -> None:
    """
    Register the Tavily search tool onto the FastMCP instance.
    """
    
    @mcp.tool(
        name="tavily_search",
        description="Search the web using Tavily. Use when document knowledge doesn't give a result or not relevant result."
    )
    def tavily_search(query: str) -> str:
        tavily_client = TavilyClient(api_key=mcp_settings.TAVILY_API_KEY)
        response = tavily_client.search(query)
        return str(response)
