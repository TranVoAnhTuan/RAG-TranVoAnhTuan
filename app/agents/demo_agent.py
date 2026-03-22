from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
# from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from langchain.tools import tool
from langchain.agents import AgentState, create_agent
from langgraph.checkpoint.memory import InMemorySaver  

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.services.rag_service import RAGService
from app.core.config import settings

rag_service = RAGService()

@tool
async def search_document_knowledge(query: str) -> str:
    """
    Công cụ tra cứu cơ sở dữ liệu tài liệu. 
    LƯU Ý QUAN TRỌNG: Chỉ truyền vào MỘT ý định hoặc MỘT khía cạnh duy nhất cho mỗi lần gọi. 
    Nếu câu hỏi có nhiều ý, hãy gọi công cụ này nhiều lần (có thể gọi song song) để tìm kiếm riêng biệt từng ý.
    """
    return await rag_service.retrieve_and_rerank(query)

class DemoAgent:
    def __init__(self):
        self.model = ChatOllama(
            model="llama3.2:1b",
            temperature=0,
            base_url="http://localhost:11434"
        )
        self.memory = InMemorySaver()
        self.tools = [search_document_knowledge]

        self.system_prompt = """
                    Bạn là một trợ lý thông minh. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng dựa trên tài liệu đã cung cấp.

                    BẮT BUỘC TUÂN THỦ QUY TRÌNH SAU:
                    1. Phân tích câu hỏi và tách thành các ý nhỏ (sub-questions)
                    2. Với MỖI sub-question:
                    - Gọi tool `search_document_knowledge`
                    - KHÔNG được gộp nhiều ý vào một query
                    3. Sau khi gọi tool nhiều lần, tổng hợp kết quả

                    QUY TẮC:
                    - Nếu có 2 ý → gọi tool 2 lần
                    - Nếu có 3 ý → gọi tool 3 lần
                    - Không được skip bước này
                    """

        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self.system_prompt,
            checkpointer = self.memory
        )

    async def ask(self, query: str, thread_id: str = "1") -> str:
        final_answer = ""
        config = {"configurable": {"thread_id": thread_id}}

        try:
            async for step in self.agent.astream(
                {"messages": [{"role": "user", "content": query}]},
                config=config,
                stream_mode="values",
            ):
                messages = step.get("messages", [])
                if not messages:
                    continue
                
                latest_msg = messages[-1]
                
                # Debug message
                latest_msg.pretty_print()

                # Kiểm tra tin nhắn cuối cùng từ AI
                if latest_msg.type == "ai" and latest_msg.content and not latest_msg.tool_calls:
                    final_answer = latest_msg.content
        except Exception as e:
            print(f"Error during agent execution: {e}")
            return f"Đã xảy ra lỗi: {str(e)}"

        return final_answer or "Không có câu trả lời."


