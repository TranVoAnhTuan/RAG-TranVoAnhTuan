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

# class DemoAgent:
#     def __init__(self):
#         self.llm = ChatGroq(
#             model="llama-3.3-70b-versatile",
#             temperature=0,
#             api_key=settings.API_KEY 
#         )
#         self.tools = [search_document_knowledge]
        
#         self.prompt = ChatPromptTemplate.from_messages([
#             ("system", 
#              "Bạn là một trợ lý thông minh. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng dựa trên tài liệu đã cung cấp. "
#              "BẠN BẮT BUỘC PHẢI DÙNG CÔNG CỤ 'search_document_knowledge' để tìm thông tin trước khi trả lời.\n\n"
#              "CHIẾN LƯỢC TÌM KIẾM ĐA Ý ĐỊNH: "
#              "Nếu câu hỏi của người dùng chứa nhiều ý, nhiều khía cạnh hoặc nhiều chủ đề khác nhau, "
#              "BẠN PHẢI TỰ ĐỘNG CHIA NHỎ CÂU HỎI VÀ GỌI CÔNG CỤ NHIỀU LẦN cho từng ý đó. "
#              "Tuyệt đối không gộp chung nhiều ý phức tạp vào một lần gọi duy nhất.\n\n"
#              "Sau khi đã thu thập đủ thông tin từ các lần tìm kiếm, hãy tổng hợp lại thành câu trả lời chính xác, mạch lạc bằng tiếng Việt. "
#              "Nếu kết quả tìm kiếm không có thông tin, hãy nói rõ là tài liệu không đề cập đến."),
#             ("user", "{input}"),
#             MessagesPlaceholder(variable_name="agent_scratchpad"),
#         ])
        
#         self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
#         self.executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

#         self.prompt = """

#                     """

#     def ask(self, query: str) -> str:
#         response = self.executor.invoke({"input": query})
#         return response["output"]



class DemoAgent:
    def __init__(self):
        self.model = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=settings.API_KEY
        )

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
            checkpointer=InMemorySaver(),

        )

    def ask(self, query: str) -> str:
        final_answer = ""

        for step in self.agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            for msg in step["messages"]:
                msg.pretty_print()

                if msg.type == "ai" and msg.content:
                    final_answer = msg.content

        return final_answer or "Không có câu trả lời."
    


