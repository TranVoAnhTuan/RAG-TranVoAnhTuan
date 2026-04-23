import asyncio
import gc
import json
import logging
import re
import traceback
import uuid
from typing import Any

from app.core.config import settings
from app.db.mongo_db import mongo_db
from langchain.agents import create_agent, AgentState
from langchain.messages import RemoveMessage
from langchain_core.globals import set_llm_cache, get_llm_cache
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_redis import RedisCache
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command

logger = logging.getLogger(__name__)

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

def _parse_gemma_tool_calls(content: str) -> list[dict]:
    calls = []
    for m in re.finditer(
        r'<\|tool_call\|?>call:([a-zA-Z_][a-zA-Z0-9_]*)\{(.*?)\}<\|?tool_call\|?>',
        content, re.DOTALL,
    ):
        name, raw_args = m.group(1), m.group(2)
        args = {
            am.group(1): am.group(2)
            for am in re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*):<\|"\|>(.*?)<\|"\|>', raw_args, re.DOTALL)
        }
        if name:
            calls.append({"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:12]}"})
    return calls

class _GemmaToolFix:
    def __init__(self, model):
        self._model = model

    def bind_tools(self, tools, **kwargs):
        self._model = self._model.bind_tools(tools, **kwargs)
        return self

    async def ainvoke(self, input, config=None, **kwargs):
        res = await self._model.ainvoke(input, config=config, **kwargs)
        if (
            not getattr(res, "tool_calls", None)
            and hasattr(res, "content")
            and isinstance(res.content, str)
            and "<|tool_call" in res.content
        ):
            parsed = _parse_gemma_tool_calls(res.content)
            if parsed:
                logger.info(f"🛠️ [GemmaFix] Converted {len(parsed)} tag-based tool call(s) to structured format.")
                res.tool_calls = parsed
                res.content = ""
        return res

    def __getattr__(self, name):
        return getattr(self._model, name)

class DemoAgent:
    def __init__(self):
        try:
            if get_llm_cache() is None:
                set_llm_cache(RedisCache(redis_url=settings.REDIS_URL))
                logger.info(f"✅ LLM Cache initialized at {settings.REDIS_URL}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM Cache: {e}")

        self.model = _GemmaToolFix(ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0,
            base_url=settings.BASE_URL,
            api_key="none",
            timeout=60,
            max_retries=2,
        ))
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
        )
        logger.info("🚀 DemoAgent is ready.")

    async def disconnect_mcp(self) -> None:
        if self._mcp_client is not None:
            self._mcp_client = None
            logger.info("🔌 MCP client disconnected.")

    async def ask(self, query: str, thread_id: str = "1", topic: str = "General") -> dict:
        if self.agent is None:
            return {"response": "Agent not initialized.", "citations": []}

        config = {"configurable": {"thread_id": thread_id}}
        dynamic_topic_instruction = (
            f"CRITICAL: The user has selected the topic '{topic}'. "
            f"You MUST pass '{topic}' to the `topic` argument when calling `search_document_knowledge`."
        )

        try:
            final_state = await self.agent.ainvoke(
                {
                    "messages": [
                        {"role": "system", "content": dynamic_topic_instruction},
                        {"role": "human", "content": query},
                    ],
                },
                config=config,
                version="v2",
            )

            if hasattr(final_state, "interrupts") and final_state.interrupts:
                interrupt = final_state.interrupts[0].value
                return {
                    "interrupt": True,
                    "action_requests": interrupt.get("action_requests", [])
                }

            messages = final_state.value.get("messages", []) if hasattr(final_state, "value") else final_state.get("messages", [])
            if not messages:
                return {"response": "No answer found.", "citations": []}

            final_message = messages[-1]
            final_answer = ""
            if hasattr(final_message, "content") and final_message.content:
                content = final_message.content
                final_answer = "".join(b.get("text", "") for b in content if isinstance(b, dict)) if isinstance(content, list) else content

            if isinstance(final_answer, str):
                match = re.search(r"\{.*\}", final_answer, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(0))
                    except json.JSONDecodeError:
                        pass
                return {"response": final_answer, "citations": []}

            return final_answer if isinstance(final_answer, dict) else {"response": final_answer or "No answer.", "citations": []}

        except Exception as e:
            logger.error(f"Agent execution error: {e}\n{traceback.format_exc()}")
            return {"response": f"An error occurred: {str(e)}", "citations": []}

    async def resume(self, thread_id: str, decision: str) -> dict:
        if self.agent is None:
            return {"response": "Agent not initialized.", "citations": []}

        config = {"configurable": {"thread_id": thread_id}}
        try:
            final_state = await self.agent.ainvoke(
                Command(resume={"decisions": [{"type": decision}]}),
                config=config,
                version="v2",
            )

            if hasattr(final_state, "interrupts") and final_state.interrupts:
                interrupt = final_state.interrupts[0].value
                return {
                    "interrupt": True,
                    "action_requests": interrupt.get("action_requests", [])
                }

            messages = final_state.value.get("messages", []) if hasattr(final_state, "value") else final_state.get("messages", [])
            if not messages:
                return {"response": "No answer found.", "citations": []}

            final_message = messages[-1]
            final_answer = ""
            if hasattr(final_message, "content") and final_message.content:
                content = final_message.content
                final_answer = "".join(b.get("text", "") for b in content if isinstance(b, dict)) if isinstance(content, list) else content

            if isinstance(final_answer, str):
                match = re.search(r"\{.*\}", final_answer, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(0))
                    except json.JSONDecodeError:
                        pass
                return {"response": final_answer, "citations": []}

            return final_answer if isinstance(final_answer, dict) else {"response": final_answer or "No answer.", "citations": []}

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
                
                citations = []
                if role == "assistant" and isinstance(content, str) and "{" in content:
                    match = re.search(r"\{.*\}", content, re.DOTALL)
                    if match:
                        try:
                            data = json.loads(match.group(0))
                            content, citations = data.get("response", content), data.get("citations", [])
                        except: pass
                processed.append({"role": role, "content": content, "citations": citations})
            return processed
        except Exception as e:
            logger.error(f"Error fetching messages: {e}")
            return []