from fastmcp import FastMCP
from pydantic import BaseModel, Field


class Citation(BaseModel):
    Header_1: str = Field(description="The primary source document or header level 1")
    Header_2: str = Field(description="The secondary header or section")
    file_url: str = Field(description="The URL or file path to the source document")


def register_tools(mcp: FastMCP) -> None:
    """
    Register the final answer submission tool onto the FastMCP instance.
    """

    @mcp.tool(
        name="submit_final_answer",
        description="Submit your final response to the user. You MUST call this tool to respond."
    )
    def submit_final_answer(
        response: str = Field(description="The natural language answer to the user"),
        citations: list[Citation] = Field(description="List of document citations used in the answer")
    ) -> str:
        # We don't actually do anything here. The LLM calling this tool is what matters.
        # The LangGraph execution will intercept this tool call's arguments.
        return "FINAL_ANSWER_SUBMITTED"
