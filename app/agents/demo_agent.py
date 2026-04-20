"""
DemoAgent — Agentic RAG orchestrator (MCP Client).

This module is now a *thin* MCP client. All tool execution logic and the
system prompt live in the separate FastMCP server (mcp_server/).

Lifecycle
─────────
  DemoAgent is created at import time but the agent graph is NOT built until
  `await agent_instance.connect_mcp()` is called (done in FastAPI lifespan).
  `await agent_instance.disconnect_mcp()` must be called on shutdown.

Communication with MCP server
──────────────────────────────
  Transport : Streamable HTTP  →  http://<MCP_SERVER_URL>/mcp
  Library   : langchain-mcp-adapters (MultiServerMCPClient)
  At startup the client:
    1. Lists tools from the MCP server  →  used by create_agent()
    2. Fetches rag_system_prompt        →  used as the agent's system_prompt
"""

import json
import re
import gc
import traceback
import uuid
from typing import Any

from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.mongodb import MongoDBSaver

from app.core.config import settings
from app.db.mongo_db import mongo_db
import asyncio

# ── Memory trimming middleware ─────────────────────────────────────────────────

@before_model
def trim_messages(state: AgentState, runtime: Any) -> dict | None:
    """
    Balanced memory trimming:
    - Keeps System prompt
    - Keeps ALL User messages
    - Keeps ALL final Assistant JSON answers
    - Keeps ONLY the MOST RECENT ToolMessage (latest search result)
    - Removes all old ToolMessages to prevent memory bloat
    """
    messages = state["messages"]
    print("\n=== ALL MESSAGE TYPES ===")
    for i, msg in enumerate(messages):
        msg_type = getattr(msg, "type", None)
        msg_class = type(msg).__name__
        content_preview = str(msg)[:100] if hasattr(msg, "__str__") else ""
        print(f"{i}: type='{msg_type}' | class={msg_class}")
        print(f"   Preview: {content_preview}\n")

    if len(messages) <= 8:
        return None

    important = []
    last_tool_message = None

    for msg in messages:
        msg_type = getattr(msg, "type", None)

        if msg_type == "system":
            important.append(msg)
            continue

        if msg_type == "human":           # Keep every user question
            important.append(msg)
            continue

        if msg_type == "ai":
            # Keep only final answers (no tool_calls)
            if not getattr(msg, "tool_calls", None):
                important.append(msg)
            continue

        if msg_type == "tool":            # Keep ONLY the latest tool result
            last_tool_message = msg

    if last_tool_message:
        important.append(last_tool_message)

    # Limit: System + max 5 recent User/Assistant pairs + 1 latest tool
    if len(important) > 13:
        system = important[0]
        recent = important[-12:]
        important = [system] + recent

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *important,
        ]
    }


# ── Gemma Tool Call Parser ─────────────────────────────────────────────────────
# vLLM doesn't translate Gemma's native <|tool_call> tags into OpenAI tool_calls.
# This wrapper intercepts the raw text and converts it to structured tool_calls.

def _parse_gemma_tool_calls(content: str) -> list[dict]:
    """Parse <|tool_call>call:name{key:<|"|>val<|"|>}<tool_call|> into tool_calls."""
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
    """Thin wrapper: patches tool_calls onto AIMessages when vLLM returns raw tags."""

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
                print(f"🛠️ [GemmaFix] Converted {len(parsed)} tag-based tool call(s) to structured format.")
                res.tool_calls = parsed
                res.content = ""  # Clear text so agent doesn't treat it as a final answer
        return res

    def __getattr__(self, name):
        return getattr(self._model, name)


# ── DemoAgent ──────────────────────────────────────────────────────────────────

class DemoAgent:
    """
    Agentic RAG orchestrator that connects to the FastMCP server for tools
    and the system prompt.
    """

    def __init__(self):
        self.model = _GemmaToolFix(ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0,
            base_url=settings.BASE_URL,
            api_key="none",
            timeout=60,
            max_retries=2,
        ))
        self.memory = MongoDBSaver(mongo_db.client)

        # Populated by connect_mcp()
        self._mcp_client: MultiServerMCPClient | None = None
        self.agent = None
        self.system_prompt: str = ""

    # ── MCP lifecycle ──────────────────────────────────────────────────────────

    async def connect_mcp(self) -> None:
        """
        Open a persistent connection to the FastMCP server, discover tools,
        fetch the system prompt, and build the LangGraph agent.

        Called once from FastAPI's lifespan startup hook.
        """
        print(f"🔌 Connecting to MCP server at {settings.MCP_SERVER_URL} …")

        self._mcp_client = MultiServerMCPClient(
            {
                "rag_tools": {
                    "url": settings.MCP_SERVER_URL,
                    "transport": "streamable_http",
                }
            }
        )

        # 1. Discover tools exposed by the MCP server
        # In langchain-mcp-adapters >= 0.1.0, get_tools() is async and handles connection
        # We wrap this in a retry loop because mcp-server needs time to load heavy models into VRAM
        
        max_retries = 60
        for i in range(max_retries):
            try:
                tools = await self._mcp_client.get_tools()
                print(f"✅ MCP tools loaded: {[t.name for t in tools]}")
                break
            except Exception as e:
                if i == max_retries - 1:
                    print(f"❌ Failed to connect to MCP server after {max_retries} attempts.")
                    raise e
                print(f"⏳ Waiting for MCP server to start... (Attempt {i+1}/{max_retries}): {e}")
                await asyncio.sleep(4)

        # 2. Fetch the system prompt from the MCP server
        # We need to use the session manager context or directly access it.
        # Let's try directly accessing the session, but wait, if we need a context manager for the session...
        # The error suggests: async with client.session("server_name") as session:
        async with self._mcp_client.session("rag_tools") as session:
            prompt_result = await session.get_prompt("rag_system_prompt")
        self.system_prompt = prompt_result.messages[0].content.text
        print("✅ System prompt fetched from MCP server.")

        # 3. Build the LangGraph agent with the remote tools & prompt
        self.agent = create_agent(
            model=self.model,
            tools=tools,
            system_prompt=self.system_prompt,
            checkpointer=self.memory,
            middleware=[trim_messages],
        )
        print("🚀 DemoAgent is ready.")

    async def disconnect_mcp(self) -> None:
        """
        Close the MCP connection gracefully.
        Called from FastAPI's lifespan shutdown hook.
        """
        if self._mcp_client is not None:
            self._mcp_client = None
            print("🔌 MCP client disconnected.")

    # ── Main interaction method ────────────────────────────────────────────────

    async def ask(self, query: str, thread_id: str = "1", topic: str = "General") -> dict:
        """Main method to interact with the Agent."""
        if self.agent is None:
            return {"response": "Agent not initialized. MCP connection may have failed.", "citations": []}

        config = {"configurable": {"thread_id": thread_id}}

        dynamic_topic_instruction = (
            f"CRITICAL: The user has selected the topic '{topic}'. "
            f"You MUST pass '{topic}' to the `topic` argument when calling "
            f"`search_document_knowledge`."
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
            )

            messages = final_state.get("messages", [])
            if not messages:
                return {"response": "No answer found.", "citations": []}

            final_message = messages[-1]

            # ── Debug logging ──────────────────────────────────────────────────
            print("\n" + "=" * 80)
            print("🔍 DEBUG REASONING - FINAL MESSAGE")
            print(f"Type: {type(final_message).__name__}")
            print("\n--- Full final_message ---")
            print(final_message)
            print("\n--- .content ---")
            print(repr(final_message.content))
            print("\n--- .additional_kwargs ---")
            print(final_message.additional_kwargs)
            reasoning = final_message.additional_kwargs.get("reasoning_content")
            if reasoning:
                print(f"\n--- REASONING_CONTENT (len: {len(reasoning)}) ---")
                print(repr(reasoning[:1500] + "..." if len(reasoning) > 1500 else reasoning))
            else:
                print("\n--- No reasoning_content ---")
            print("\n--- .response_metadata ---")
            print(final_message.response_metadata)
            print("=" * 80 + "\n")
            # ──────────────────────────────────────────────────────────────────

            final_answer = ""
            if hasattr(final_message, "content") and final_message.content:
                content = final_message.content
                if isinstance(content, list):
                    final_answer = "".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict)
                    )
                else:
                    final_answer = content

            print("\n" + "=" * 60)
            print("🔍 FINAL MESSAGE TYPE:", type(final_message).__name__)
            print("🔍 FINAL MESSAGE CONTENT (RAW):")
            print(final_answer)
            print("=" * 60 + "\n")

            if isinstance(final_answer, str):
                match = re.search(r"\{.*\}", final_answer, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(0))
                    except json.JSONDecodeError:
                        return {"response": final_answer, "citations": []}
                return {"response": final_answer, "citations": []}

            if isinstance(final_answer, dict):
                return final_answer

            return {"response": final_answer or "No answer.", "citations": []}

        except Exception as e:
            print(f"Agent execution error: {e}")
            traceback.print_exc() # Show full error details in logs
            return {"response": f"An error occurred: {str(e)}", "citations": []}