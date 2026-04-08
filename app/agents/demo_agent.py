import json
import re
import os
from typing import Any, Dict

from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from app.services.rag_service import RAGService
from app.core.config import settings
from app.agents.prompt import SYSTEM_PROMPT

rag_service = RAGService()

class CustomAgentState(AgentState):
    important_metadata: Dict[str, Any]

@before_model
def trim_messages(state: CustomAgentState, runtime: Any) -> dict | None:
    """
    Keep the context concise by retaining only the system message (0) and the 4 most recent messages in the conversation history.
    """
    messages = state["messages"]
    
    if len(messages) <= 5:
        return None
    
    first_msg = messages[0]
    recent_messages = messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

@tool
async def search_document_knowledge(query: str) -> str:
    """
    Search for information in the document knowledge base.
    IMPORTANT GUIDELINE: If called a 2nd or 3rd time, the query must represent a completely different approach or a broader topic.
    """
    return await rag_service.retrieve_and_rerank(query)

class DemoAgent:
    def __init__(self):
        self.model = ChatOllama(
            model=settings.LLM_MODEL, 
            temperature=0,
            base_url=settings.BASE_URL,
            keep_alive="0s",
            format="json"
        )
        
        self.memory = InMemorySaver()
        self.tools = [search_document_knowledge]
        self.system_prompt = SYSTEM_PROMPT

        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self.system_prompt,
            checkpointer=self.memory,
            state_schema=CustomAgentState,
            middleware=[trim_messages]
        )

    async def ask(self, query: str, thread_id: str = "1") -> dict:
        """Main method to interact with the Agent."""
        config = {"configurable": {"thread_id": thread_id}}

        try:
            final_state = await self.agent.ainvoke(
                {
                    "messages": [{"role": "user", "content": query}],
                    "important_metadata": {} 
                },
                config=config
            )

            messages = final_state.get("messages", [])
            if not messages:
                return {"response": "No answer found.", "citations": []}
            
            final_message = messages[-1]
            final_answer = ""
          
            if hasattr(final_message, "content") and final_message.content:
                final_answer = final_message.content

            if isinstance(final_answer, str):
                match = re.search(r'\{.*\}', final_answer, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        return {"response": final_answer, "citations": []}
                else:
                    return {"response": final_answer, "citations": []}

            if isinstance(final_answer, dict):
                return final_answer

            return {"response": final_answer or "No answer.", "citations": []}

        except Exception as e:
            print(f"Agent execution error: {e}")
            return {"response": f"An error occurred: {str(e)}", "citations": []}