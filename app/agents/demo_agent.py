"""Agentic RAG orchestrator.

Design Patterns
---------------
- **Facade**: ``DemoAgent`` provides a single interface over MCP tools,
  LangGraph agent, Redis cache, and MongoDB checkpointing.
- **Chain of Responsibility**: ``_extract_response`` delegates to a
  chain of extraction handlers, each returning a result or ``None``
  to pass to the next handler.
- **Singleton**: instantiated once in ``app/api/routes.py``.
"""

import asyncio
import hashlib
import json
import logging
import re
import traceback
from typing import Any

import redis.asyncio as redis
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
        try:
            self.app_cache: redis.Redis | None = redis.from_url(settings.REDIS_URL, decode_responses=True)
            logger.info(f"✅ App Cache initialized at {settings.REDIS_URL}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize App Cache: {e}")
            self.app_cache = None

        self.model = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0,
            base_url=settings.BASE_URL,
            api_key="none",
            timeout=60,
            max_retries=2,
            max_tokens=50000,
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
    # Response extraction — Chain of Responsibility
    # ------------------------------------------------------------------

    @staticmethod
    def _find_turn_messages(messages: list) -> list:
        """Scope to the current turn: everything after the last human message."""
        last_human_idx = -1
        for i, m in enumerate(messages):
            if getattr(m, "type", "") == "human":
                last_human_idx = i
        return messages[last_human_idx + 1 :] if last_human_idx >= 0 else messages

    @staticmethod
    def _collect_tools_used(turn_messages: list) -> list[dict]:
        """Collect non-final tool calls made during this turn."""
        tools_used: list[dict] = []
        for m in turn_messages:
            if getattr(m, "type", "") == "ai" and getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    name = tc.get("name", "")
                    if name == "submit_final_answer":
                        continue
                    tools_used.append({"name": name, "args": tc.get("args", {})})
        return tools_used

    @staticmethod
    def _extract_from_submit_tool(turn_messages: list, tools_used: list[dict]) -> dict | None:
        """Primary handler: extract from submit_final_answer tool call."""
        for m in reversed(turn_messages):
            if getattr(m, "type", "") == "ai" and getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    if tc.get("name") == "submit_final_answer":
                        args = tc.get("args", {})
                        return {
                            "response": str(args.get("response", "No answer found.")),
                            "citations": list(args.get("citations", [])),
                            "tools_used": tools_used,
                        }
        return None

    @staticmethod
    def _extract_raw_text(turn_messages: list) -> str:
        """Get raw text content from the final AI message."""
        for m in reversed(turn_messages):
            if getattr(m, "type", "") != "ai":
                continue
            content = getattr(m, "content", "")
            if isinstance(content, list):
                content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
            if content and content.strip():
                return content
        return ""

    @staticmethod
    def _extract_auto_citations(turn_messages: list) -> list[dict]:
        """Extract citations from search_document_knowledge tool responses."""
        auto_citations: list[dict] = []
        for m in reversed(turn_messages):
            if getattr(m, "type", "") == "tool" and getattr(m, "name", "") == "search_document_knowledge":
                tool_text = m.content if isinstance(m.content, str) else str(m.content)
                for meta_match in re.finditer(r"METADATA:\s*(\{.*?\})", tool_text):
                    try:
                        meta = json.loads(meta_match.group(1))
                        if "file_url" in meta:
                            auto_citations.append(
                                {
                                    "Header_1": meta.get("Header_1", "Source"),
                                    "Header_2": meta.get("Header_2", ""),
                                    "file_url": meta.get("file_url", ""),
                                }
                            )
                    except json.JSONDecodeError:
                        pass
                break
        return auto_citations

    @staticmethod
    def _parse_json_response(raw_text: str, auto_citations: list[dict], tools_used: list[dict]) -> dict | None:
        """Try direct JSON parse, then regex extraction from raw text."""
        # Attempt 1: direct JSON parse
        try:
            parsed = json.loads(raw_text)
            citations = parsed.get("citations")
            if citations is None:
                citations = auto_citations
            return {
                "response": str(parsed.get("response", raw_text)),
                "citations": list(citations),
                "tools_used": tools_used,
            }
        except json.JSONDecodeError:
            pass

        # Attempt 2: regex extraction (model sometimes wraps JSON in text)
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                citations = parsed.get("citations")
                if citations is None:
                    citations = auto_citations
                return {
                    "response": str(parsed.get("response", raw_text)),
                    "citations": list(citations),
                    "tools_used": tools_used,
                }
            except json.JSONDecodeError:
                pass

        return None

    async def _extract_response(self, messages: list) -> dict:
        """Extract structured response from agent messages.

        Chain of Responsibility: each handler returns a result or ``None``
        to pass to the next handler in priority order:
        1. submit_final_answer tool call (structured)
        2. JSON parse from raw AI text
        3. Plain text with auto-extracted citations
        """
        turn_messages = self._find_turn_messages(messages)
        tools_used = self._collect_tools_used(turn_messages)

        # Handler 1: submit_final_answer tool call
        result = self._extract_from_submit_tool(turn_messages, tools_used)
        if result is not None:
            return result

        # Handler 2 & 3: raw text → JSON parse → plain text fallback
        raw_text = self._extract_raw_text(turn_messages)
        if not raw_text:
            return {"response": "No answer found.", "citations": [], "tools_used": tools_used}

        auto_citations = self._extract_auto_citations(turn_messages)

        result = self._parse_json_response(raw_text, auto_citations, tools_used)
        if result is not None:
            return result

        # Final fallback: plain text with auto-citations
        return {"response": raw_text, "citations": auto_citations, "tools_used": tools_used}

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
    # Cache helpers
    # ------------------------------------------------------------------

    async def _check_cache(self, query: str, topic: str) -> tuple[str | None, dict | None]:
        """Check Redis application cache. Returns (cache_key, cached_result) or (key, None)."""
        if self.app_cache is None:
            return None, None

        normalized_query = query.strip().lower()
        hash_str = hashlib.md5(f"{topic}:{normalized_query}".encode()).hexdigest()
        cache_key = f"app_cache:{hash_str}"

        try:
            cached_res = await self.app_cache.get(cache_key)
            if cached_res:
                logger.info("⚡ Application Cache HIT! Returning instantly.")
                return cache_key, json.loads(cached_res)
        except Exception as e:
            logger.error(f"App Cache read error: {e}")

        return cache_key, None

    async def _write_cache(self, cache_key: str, result: dict) -> None:
        """Write a result to the Redis application cache."""
        if self.app_cache is None or cache_key is None:
            return
        try:
            ttl = getattr(settings, "CACHE_TTL_SECONDS", 86400)
            await self.app_cache.setex(cache_key, ttl, json.dumps(result))
        except Exception as e:
            logger.error(f"App Cache write error: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ask(self, query: str, thread_id: str = "1", topic: str = "General") -> dict:
        """Send a user query to the agent and return a structured response."""
        if self.agent is None:
            return {"response": "Agent not initialized.", "citations": []}

        # Check application-level cache
        cache_key, cached_result = await self._check_cache(query, topic)
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

            answer_dict = await self._extract_response(messages)

            await self._write_cache(cache_key, answer_dict)
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

            return await self._extract_response(messages)

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
        try:
            db = mongo_db.client["checkpointing_db"]
            pipeline = [
                {"$sort": {"_id": -1}},
                {"$group": {"_id": "$thread_id", "latest_id": {"$first": "$_id"}}},
                {"$sort": {"latest_id": -1}},
                {"$limit": 50},
            ]
            results = list(db["checkpoints"].aggregate(pipeline))
            return [
                {"thread_id": res["_id"], "title": f"Conversation {res['_id'][:8]}"}
                for res in results
                if res["_id"] != "default_thread"
            ]
        except Exception as e:
            logger.error(f"Error fetching threads: {e}")
            return []

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete all checkpoint data for a given thread."""
        try:
            db = mongo_db.client["checkpointing_db"]
            db["checkpoints"].delete_many({"thread_id": thread_id})
            db["checkpoint_writes"].delete_many({"thread_id": thread_id})
            return True
        except Exception as e:
            logger.error(f"Error deleting thread {thread_id}: {e}")
            return False

    async def get_thread_messages(self, thread_id: str) -> list[dict]:
        """Retrieve human/assistant messages for a thread, parsing any JSON content."""
        try:
            state = self.agent.get_state({"configurable": {"thread_id": thread_id}})
            messages = state.values.get("messages", []) if state.values else []
            processed: list[dict] = []
            for msg in messages:
                if msg.type not in ("human", "ai"):
                    continue
                role = "user" if msg.type == "human" else "assistant"
                content = msg.content

                if role == "assistant":
                    if not content or (hasattr(msg, "additional_kwargs") and msg.additional_kwargs.get("tool_calls")):
                        continue
                    if (
                        str(content)
                        .strip()
                        .upper()
                        .startswith(
                            ("CRITICAL:", "SEARCH RESULTS", "TOOL RESPONSE", "INTERNAL:", "THOUGHT:", "OBSERVATION:")
                        )
                    ):
                        continue

                if isinstance(content, list):
                    content = "".join(b.get("text", "") for b in content if isinstance(b, dict))

                # Try to parse JSON from older messages that used the JSON format
                citations: list[dict] = []
                if role == "assistant" and isinstance(content, str) and "{" in content:
                    try:
                        data = json.loads(content)
                        content = data.get("response", content)
                        citations = data.get("citations", [])
                    except Exception:
                        pass
                processed.append({"role": role, "content": content, "citations": citations})
            return processed
        except Exception as e:
            logger.error(f"Error fetching messages: {e}")
            return []
