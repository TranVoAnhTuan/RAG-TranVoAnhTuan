import asyncio
import json
import logging
import re
import traceback
from typing import Any

from app.core.config import settings
from app.db.mongo_db import mongo_db
from langchain.agents import create_agent, AgentState
from langchain.messages import RemoveMessage
import hashlib
import redis.asyncio as redis
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents.middleware import before_model, HumanInTheLoopMiddleware
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command


logger = logging.getLogger(__name__)

@before_model
def trim_messages(state: AgentState, runtime: Any) -> dict | None:
    messages = state["messages"]
    if len(messages) <= 12:
        return None

    system_msg = messages[0] if messages and messages[0].type == "system" else None
    recent_messages = messages[-10:]
    
    important = []
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


class DemoAgent:
    def __init__(self):
        try:
            self.app_cache = redis.from_url(settings.REDIS_URL, decode_responses=True)
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
        ).bind(parallel_tool_calls=False)

        self.memory = MongoDBSaver(mongo_db.client, db_name="checkpointing_db")
        self._mcp_client: MultiServerMCPClient | None = None
        self.agent = None
        self.system_prompt: str = ""

    async def connect_mcp(self) -> None:
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
        for i in range(max_retries):
            try:
                tools = await self._mcp_client.get_tools()
                logger.info(f"✅ MCP tools loaded: {[t.name for t in tools]}")
                break
            except Exception as e:
                if i == max_retries - 1:
                    logger.error(f"❌ Failed to connect to MCP server: {e}")
                    raise e
                logger.info(f"⏳ Waiting for MCP server... (Attempt {i+1}/{max_retries})")
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
                )
            ],
        )
        logger.info("🚀 DemoAgent is ready.")

    async def disconnect_mcp(self) -> None:
        if self._mcp_client is not None:
            self._mcp_client = None
            logger.info("🔌 MCP client disconnected.")

    async def _extract_response(self, messages: list) -> dict:
        """Extract structured response from agent messages via JSON parsing."""
        # Scope to current turn only: find the last human message and slice after it
        last_human_idx = -1
        for i, m in enumerate(messages):
            if getattr(m, "type", "") == "human":
                last_human_idx = i
        turn_messages = messages[last_human_idx + 1:] if last_human_idx >= 0 else messages

        # Collect tool calls made during this turn
        tools_used = []
        for m in turn_messages:
            if getattr(m, "type", "") == "ai" and getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    name = tc.get("name", "")
                    if name == "submit_final_answer":
                        continue
                    tools_used.append({"name": name, "args": tc.get("args", {})})

        # 1. Primary approach: Intercept submit_final_answer tool call
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

        # Get raw text from the final AI message (Emergency Fallback)
        raw_text = ""
        for m in reversed(turn_messages):
            if getattr(m, "type", "") != "ai":
                continue
            content = getattr(m, "content", "")
            if isinstance(content, list):
                content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
            if content and content.strip():
                raw_text = content
                break

        if not raw_text:
            return {"response": "No answer found.", "citations": [], "tools_used": tools_used}

        # Extract citations from tool responses
        auto_citations = []
        for m in reversed(turn_messages):
            if getattr(m, "type", "") == "tool" and getattr(m, "name", "") == "search_document_knowledge":
                tool_text = m.content if isinstance(m.content, str) else str(m.content)
                for meta_match in re.finditer(r'METADATA:\s*(\{.*?\})', tool_text):
                    try:
                        meta = json.loads(meta_match.group(1))
                        if "file_url" in meta:
                            auto_citations.append({
                                "Header_1": meta.get("Header_1", "Source"),
                                "Header_2": meta.get("Header_2", ""),
                                "file_url": meta.get("file_url", ""),
                            })
                    except json.JSONDecodeError:
                        pass
                break

        # 1. Try direct JSON parse (instant)
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

        # 2. Try regex extraction — model sometimes wraps JSON in text (instant)
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
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

        # 3. Model responded in natural language — use raw text + tool citations
        return {"response": raw_text, "citations": auto_citations, "tools_used": tools_used}

    async def ask(self, query: str, thread_id: str = "1", topic: str = "General") -> dict:
        if self.agent is None:
            return {"response": "Agent not initialized.", "citations": []}

        # ── APPLICATION-LEVEL CACHING ────────────────────────────────────────────────
        cache_key = None
        if self.app_cache is not None:
            normalized_query = query.strip().lower()
            hash_str = hashlib.md5(f"{topic}:{normalized_query}".encode("utf-8")).hexdigest()
            cache_key = f"app_cache:{hash_str}"
            try:
                cached_res = await self.app_cache.get(cache_key)
                if cached_res:
                    logger.info("⚡ Application Cache HIT! Returning instantly.")
                    return json.loads(cached_res)
            except Exception as e:
                logger.error(f"App Cache read error: {e}")
        # ─────────────────────────────────────────────────────────────────────────────

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

            if hasattr(final_state, "interrupts") and final_state.interrupts:
                interrupt = final_state.interrupts[0].value
                return {
                    "response": "Approval required.",
                    "citations": [],
                    "interrupt": True,
                    "action_requests": interrupt.get("action_requests", []),
                }

            messages = final_state.value.get("messages", []) if hasattr(final_state, "value") else final_state.get("messages", [])
            if not messages:
                return {"response": "No answer found.", "citations": []}

            answer_dict = await self._extract_response(messages)
            
            # Save to App Cache if we got a real response
            if self.app_cache is not None and cache_key is not None:
                try:
                    ttl = getattr(settings, "CACHE_TTL_SECONDS", 86400)
                    await self.app_cache.setex(cache_key, ttl, json.dumps(answer_dict))
                except Exception as e:
                    logger.error(f"App Cache write error: {e}")
                    
            return answer_dict

        except asyncio.TimeoutError:
            logger.error("Agent execution timed out after 120s")
            return {"response": "Request timed out. Please try a simpler question.", "citations": []}
        except Exception as e:
            logger.error(f"Agent execution error: {e}\n{traceback.format_exc()}")
            return {"response": f"An error occurred: {str(e)}", "citations": []}

    async def resume(self, thread_id: str, decision: str) -> dict:
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

            if hasattr(final_state, "interrupts") and final_state.interrupts:
                interrupt = final_state.interrupts[0].value
                return {
                    "response": "Approval required.",
                    "citations": [],
                    "interrupt": True,
                    "action_requests": interrupt.get("action_requests", []),
                }

            messages = final_state.value.get("messages", []) if hasattr(final_state, "value") else final_state.get("messages", [])
            if not messages:
                return {"response": "No answer found.", "citations": []}

            return await self._extract_response(messages)

        except asyncio.TimeoutError:
            logger.error("Agent resume timed out after 120s")
            return {"response": "Request timed out. Please try a simpler question.", "citations": []}
        except Exception as e:
            logger.error(f"Agent resume error: {e}\n{traceback.format_exc()}")
            return {"response": f"An error occurred: {str(e)}", "citations": []}

    async def get_all_threads(self) -> list[dict]:
        try:
            db = mongo_db.client["checkpointing_db"]
            pipeline = [
                {"$sort": {"_id": -1}},
                {"$group": {"_id": "$thread_id", "latest_id": {"$first": "$_id"}}},
                {"$sort": {"latest_id": -1}},
                {"$limit": 50}
            ]
            results = list(db["checkpoints"].aggregate(pipeline))
            return [{"thread_id": res["_id"], "title": f"Conversation {res['_id'][:8]}"} for res in results if res["_id"] != "default_thread"]
        except Exception as e:
            logger.error(f"Error fetching threads: {e}")
            return []

    async def delete_thread(self, thread_id: str) -> bool:
        try:
            db = mongo_db.client["checkpointing_db"]
            db["checkpoints"].delete_many({"thread_id": thread_id})
            db["checkpoint_writes"].delete_many({"thread_id": thread_id})
            return True
        except Exception as e:
            logger.error(f"Error deleting thread {thread_id}: {e}")
            return False

    async def get_thread_messages(self, thread_id: str) -> list[dict]:
        try:
            state = self.agent.get_state({"configurable": {"thread_id": thread_id}})
            messages = state.values.get("messages", []) if state.values else []
            processed = []
            for msg in messages:
                if msg.type not in ("human", "ai"): continue
                role = "user" if msg.type == "human" else "assistant"
                content = msg.content
                if role == "assistant":
                    if not content or (hasattr(msg, "additional_kwargs") and msg.additional_kwargs.get("tool_calls")): continue
                    if str(content).strip().upper().startswith(("CRITICAL:", "SEARCH RESULTS", "TOOL RESPONSE", "INTERNAL:", "THOUGHT:", "OBSERVATION:")): continue
                
                if isinstance(content, list):
                    content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
                
                # Try to parse JSON from older messages that used the JSON format
                citations = []
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