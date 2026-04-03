from langchain_ollama import ChatOllama

from langchain.tools import tool
from langchain.agents import AgentState, create_agent
from langgraph.checkpoint.memory import InMemorySaver  
import json
import os

from app.services.rag_service import RAGService
from app.core.config import settings
from app.agents.document_info import DocumentInfo
from app.agents.prompt import SYSTEM_PROMPT
import re

rag_service = RAGService()

@tool
async def search_document_knowledge(query: str) -> str:
    """
    Searches the document knowledge base.
    CRITICAL INSTRUCTION FOR RE-SEARCHING:
    If you are calling this tool for the 2nd or 3rd time, your `query` MUST represent a completely different approach or broader topic. 
    Never use variations or synonyms of a query you already tried.
    """
    return await rag_service.retrieve_and_rerank(query)

class DemoAgent:
    def __init__(self):
        self.model = ChatOllama(
            # model="llama3.2:1b",
            model= "qwen3.5:2b",
            temperature=0,
            base_url="http://localhost:11434",
            keep_alive="0s",
            format="json"
        )
        self.memory = InMemorySaver()
        self.tools = [search_document_knowledge]

        self.system_prompt =SYSTEM_PROMPT

        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self.system_prompt,
            checkpointer = self.memory,
            # response_format= DocumentInfo
        )

    async def ask(self, query: str, thread_id: str = "1") -> dict:
        config = {"configurable": {"thread_id": thread_id}}

        try:
            # 1. Gọi ainvoke để chạy 1 mạch lấy final state
            final_state = await self.agent.ainvoke(
                {"messages": [{"role": "user", "content": query}]},
                config=config
            )

            # 2. Lấy danh sách tin nhắn từ state cuối cùng
            messages = final_state.get("messages", [])
            if not messages:
                return {"response": "No answer found.", "citations": []}
            
            # 3. Lấy tin nhắn cuối cùng (là câu trả lời của AI)
            final_message = messages[-1]
            final_answer = ""

            # 4. Trích xuất nội dung
            if hasattr(final_message, "tool_calls") and final_message.tool_calls:
                for tool in final_message.tool_calls:
                    if tool["name"] in ["DocumentInfo", "FinalAnswer"]:
                        final_answer = tool["args"]
                        break
            
            if not final_answer and hasattr(final_message, "content") and final_message.content:
                final_answer = final_message.content

            # 5. Parse JSON bằng Regex (an toàn nhất)
            if isinstance(final_answer, str):
                match = re.search(r'\{.*\}', final_answer, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        print(f"Error Decode JSON: {e}")
                        return {"response": final_answer, "citations": []}
                else:
                    return {"response": final_answer, "citations": []}

            if isinstance(final_answer, dict):
                return final_answer

            return {"response": final_answer or "No answer found.", "citations": []}

        except Exception as e:
            print(f"Error during agent execution: {e}")
            return {"response": f"An error occurred: {str(e)}", "citations": []}


