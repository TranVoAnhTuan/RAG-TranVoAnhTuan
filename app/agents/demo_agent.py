"""Agentic RAG orchestrator.

Design Patterns
---------------
- **Facade**: ``DemoAgent`` provides a single interface over MCP tools,
  LangGraph agent, Redis cache, and MongoDB checkpointing.
- **Singleton**: instantiated once in ``app/api/routes.py``.
"""

import asyncio
import logging
import traceback
from typing import Any

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, before_model
from langchain.messages import RemoveMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command

from app.core.config import settings
from app.db.mongo_db import mongo_db
from app.utils.redis_cache import AppCache
from app.utils.response_parser import ResponseParser
from app.utils.thread_repo import ThreadRepository

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Message-trimming middleware (keeps system + last 10 messages)
# ---------------------------------------------------------------------------

MAX_MESSAGES = 12
RECENT_WINDOW = 10


@before_model
def trim_messages(state: AgentState, runtime: Any) -> dict | None:
    """Trim conversation history to keep the system prompt + recent messages."""
    messages = state["messages"]
    if len(messages) <= MAX_MESSAGES:
        return None

    system_msg = messages[0] if messages and messages[0].type == "system" else None
    recent_messages = messages[-RECENT_WINDOW:]

    important: list = []
    if system_msg:
        important.append(system_msg)

    for msg in recent_messages:
        if msg not in important:
            important.append(msg)

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *important,
        ]
    }


# ---------------------------------------------------------------------------
# DemoAgent — Facade over MCP / LangGraph / Redis / MongoDB
# ---------------------------------------------------------------------------


class DemoAgent:
    """Facade that orchestrates the full RAG agent lifecycle.

    Connects to an MCP server for tools/prompts, runs queries through
    a LangGraph agent with MongoDB checkpointing, and caches results
    in Redis.
    """

    def __init__(self) -> None:
        self.app_cache = AppCache()

        self.model = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0.1,
            base_url=settings.BASE_URL,
            api_key=settings.OPENAI_KEY or "none",
            timeout=60,
            max_retries=2,
            max_tokens=13000,
        ).bind(parallel_tool_calls=False)

        self.memory = MongoDBSaver(mongo_db.client, db_name="checkpointing_db")
        self._mcp_client: MultiServerMCPClient | None = None
        self.agent = None
        self.system_prompt: str = ""

    # ------------------------------------------------------------------
    # MCP connection lifecycle
    # ------------------------------------------------------------------

    async def connect_mcp(self) -> None:
        """Connect to the MCP server, load tools, and build the agent graph."""
        logger.info(f"🔌 Connecting to MCP server at {settings.MCP_SERVER_URL} …")
        self._mcp_client = MultiServerMCPClient(
            {
                "rag_tools": {
                    "url": settings.MCP_SERVER_URL,
                    "transport": "streamable_http",
                }
            }
        )

        max_retries = 60
        for attempt in range(max_retries):
            try:
                tools = await self._mcp_client.get_tools()
                logger.info(f"✅ MCP tools loaded: {[t.name for t in tools]}")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"❌ Failed to connect to MCP server: {e}")
                    raise
                logger.info(f"⏳ Waiting for MCP server... (Attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(4)

        async with self._mcp_client.session("rag_tools") as session:
            prompt_result = await session.get_prompt("rag_system_prompt")
        self.system_prompt = prompt_result.messages[0].content.text

        self.agent = create_agent(
            model=self.model,
            tools=tools,
            system_prompt=self.system_prompt,
            checkpointer=self.memory,
            middleware=[
                trim_messages,
                HumanInTheLoopMiddleware(
                    interrupt_on={"tavily_search": True},
                    description_prefix="Tool execution pending approval",
                ),
            ],
        )
        logger.info("🚀 DemoAgent is ready.")

    async def disconnect_mcp(self) -> None:
        """Disconnect from the MCP server."""
        if self._mcp_client is not None:
            self._mcp_client = None
            logger.info("🔌 MCP client disconnected.")

    # ------------------------------------------------------------------
    # Shared helpers for ask / resume
    # ------------------------------------------------------------------

    @staticmethod
    def _get_messages_from_state(final_state: Any) -> list:
        """Extract the messages list from a LangGraph state object."""
        if hasattr(final_state, "value"):
            return final_state.value.get("messages", [])
        return final_state.get("messages", [])

    @staticmethod
    def _handle_interrupt(final_state: Any) -> dict | None:
        """Return an interrupt response dict if the agent was interrupted, else None."""
        if hasattr(final_state, "interrupts") and final_state.interrupts:
            interrupt = final_state.interrupts[0].value
            return {
                "response": "Approval required.",
                "citations": [],
                "interrupt": True,
                "action_requests": interrupt.get("action_requests", []),
            }
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ask(self, query: str, thread_id: str = "1", topic: str = "General") -> dict:
        """Send a user query to the agent and return a structured response."""
        if self.agent is None:
            return {"response": "Agent not initialized.", "citations": []}

        # Check application-level cache
        cache_key, cached_result = await self.app_cache.check_cache(query, topic)
        if cached_result is not None:
            return cached_result

        config = {"configurable": {"thread_id": thread_id}}
        dynamic_topic_instruction = (
            f"CRITICAL: The user has selected the topic '{topic}'. "
            f"You MUST pass '{topic}' to the `topic` argument when calling `search_document_knowledge`."
        )

        try:
            final_state = await asyncio.wait_for(
                self.agent.ainvoke(
                    {
                        "messages": [
                            {"role": "system", "content": dynamic_topic_instruction},
                            {"role": "human", "content": query},
                        ],
                    },
                    config=config,
                    version="v2",
                ),
                timeout=120,
            )

            interrupt_response = self._handle_interrupt(final_state)
            if interrupt_response is not None:
                return interrupt_response

            messages = self._get_messages_from_state(final_state)
            if not messages:
                return {"response": "No answer found.", "citations": []}

            answer_dict = await ResponseParser.extract_response(messages)

            await self.app_cache.write_cache(cache_key, answer_dict)
            return answer_dict

        except TimeoutError:
            logger.error("Agent execution timed out after 120s")
            return {"response": "Request timed out. Please try a simpler question.", "citations": []}
        except Exception as e:
            logger.error(f"Agent execution error: {e}\n{traceback.format_exc()}")
            return {"response": f"An error occurred: {e!s}", "citations": []}

    async def resume(self, thread_id: str, decision: str) -> dict:
        """Resume an interrupted agent execution after HITL approval/rejection."""
        if self.agent is None:
            return {"response": "Agent not initialized.", "citations": []}

        config = {"configurable": {"thread_id": thread_id}}
        try:
            final_state = await asyncio.wait_for(
                self.agent.ainvoke(
                    Command(resume={"decisions": [{"type": decision}]}),
                    config=config,
                    version="v2",
                ),
                timeout=120,
            )

            interrupt_response = self._handle_interrupt(final_state)
            if interrupt_response is not None:
                return interrupt_response

            messages = self._get_messages_from_state(final_state)
            if not messages:
                return {"response": "No answer found.", "citations": []}

            return await ResponseParser.extract_response(messages)

        except TimeoutError:
            logger.error("Agent resume timed out after 120s")
            return {"response": "Request timed out. Please try a simpler question.", "citations": []}
        except Exception as e:
            logger.error(f"Agent resume error: {e}\n{traceback.format_exc()}")
            return {"response": f"An error occurred: {e!s}", "citations": []}

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    async def get_all_threads(self) -> list[dict]:
        """Return a list of recent conversation threads from MongoDB."""
        return await ThreadRepository.get_all_threads()

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete all checkpoint data for a given thread."""
        return await ThreadRepository.delete_thread(thread_id)

    async def get_thread_messages(self, thread_id: str) -> list[dict]:
        """Retrieve human/assistant messages for a thread, parsing any JSON content."""
        if self.agent is None:
            return []
        try:
            state = self.agent.get_state({"configurable": {"thread_id": thread_id}})
            messages = state.values.get("messages", []) if state.values else []
            return ResponseParser.parse_thread_messages(messages)
        except Exception as e:
            logger.error(f"Error fetching messages: {e}")
            return []
