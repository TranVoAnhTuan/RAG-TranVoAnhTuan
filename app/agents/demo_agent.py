from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_groq import ChatGroq
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
def search_document_knowledge(query: str) -> str:
    """
    Công cụ tra cứu cơ sở dữ liệu tài liệu. 
    LƯU Ý QUAN TRỌNG: Chỉ truyền vào MỘT ý định hoặc MỘT khía cạnh duy nhất cho mỗi lần gọi. 
    Nếu câu hỏi có nhiều ý, hãy gọi công cụ này nhiều lần (có thể gọi song song) để tìm kiếm riêng biệt từng ý.
    """
    return rag_service.retrieve_and_rerank(query)

class DemoAgent:
    def __init__(self):
        self.model = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=settings.API_KEY
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

    def ask(self, query: str, thread_id: str = "1") -> str:
        final_answer = ""
        config = {"configurable": {"thread_id": thread_id}}

        # Sử dụng stream để theo dõi từng bước
        for step in self.agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            config=config,
            stream_mode="values", # Trả về toàn bộ state sau mỗi bước
        ):
            # Lấy tin nhắn mới nhất vừa được thêm vào state
            latest_msg = step["messages"][-1]
            
            # In ra để debug (chỉ in tin nhắn mới nhất của bước đó)
            latest_msg.pretty_print()

            # Kiểm tra nếu là tin nhắn của AI và có nội dung trả lời (không phải gọi tool)
            if latest_msg.type == "ai" and latest_msg.content:
                final_answer = latest_msg.content

        return final_answer or "Không có câu trả lời."


