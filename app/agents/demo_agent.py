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
    Searches the document knowledge base. 
    CRITICAL CONSTRAINT: The `query` parameter MUST contain ONLY ONE single concept or intent. Do not use complex queries with "and". 
    For multi-part questions, you MUST call this tool multiple times independently.
    """
    return await rag_service.retrieve_and_rerank(query)

class DemoAgent:
    def __init__(self):
        self.model = ChatOllama(
            # model="llama3.2:1b",
            model= "qwen3.5:2b",
            temperature=0,
            base_url="http://localhost:11434"
        )
        self.memory = InMemorySaver()
        self.tools = [search_document_knowledge]

        self.system_prompt = """
        You are a helpful and intelligent document assistant. You have access to uploaded documents, but you are also capable of natural conversation.

        STEP 1: ANALYZE THE USER'S INPUT
        Before taking any action, determine the nature of the user's query:
        1. Greetings & Small Talk: If the user says hello, asks how you are, asks about the weather, or makes general conversation, respond politely and naturally. DO NOT use the search tool.
        2. Vague or Broad Queries: If the user's question is too general, ambiguous, or lacks context (e.g., "tell me about the project", "what does it say?"), DO NOT use the search tool immediately. Ask clarifying questions to narrow down their exact need.
        3. Unknown General Facts: If the user asks a general knowledge question that you genuinely do not know, honestly admit that you don't know. Do not hallucinate.
        4. Document-Related Queries: If the user is looking for specific information likely contained in the uploaded files, proceed to use the search tool.

        STEP 2: DOCUMENT SEARCH GUIDELINES (If applicable)
        - Use the `search_document_knowledge` tool to find relevant information. 
        - The tool will return exactly 5 results. You must read and synthesize information from all 5 results to provide a comprehensive yet concise answer.
        - Be efficient: one well-crafted search is usually sufficient, but you can call the tool multiple times for multi-part questions.
        - Provide clear, accurate answers based STRICTLY on the document contents.
        - Always cite your sources with relevant headers or context if available.
        - If information isn't found in the documents after searching, say so clearly. Do not invent answers.

        When answering a document-related query:
        1. Search the documents with a focused query.
        2. Carefully read all 5 results returned by the tool.
        3. Synthesize a clear, complete, and concise answer from these results.
        4. Include source citations based on the headers or content returned.
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

                # Checking the last message of AI
                if latest_msg.type == "ai" and latest_msg.content and not latest_msg.tool_calls:
                    final_answer = latest_msg.content
        except Exception as e:
            print(f"Error during agent execution: {e}")
            return f"Đã xảy ra lỗi: {str(e)}"

        return final_answer or "Không có câu trả lời."


