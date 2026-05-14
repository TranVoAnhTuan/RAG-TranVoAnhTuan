import json
import re


class ResponseParser:
    """Handles parsing and extraction of structured responses from LangGraph/LLM messages."""

    @staticmethod
    def find_turn_messages(messages: list) -> list:
        """Scope to the current turn: everything after the last human message."""
        last_human_idx = -1
        for i, m in enumerate(messages):
            if getattr(m, "type", "") == "human":
                last_human_idx = i
        return messages[last_human_idx + 1 :] if last_human_idx >= 0 else messages

    @staticmethod
    def collect_tools_used(turn_messages: list) -> list[dict]:
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
    def extract_from_submit_tool(turn_messages: list, tools_used: list[dict]) -> dict | None:
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
    def extract_raw_text(turn_messages: list) -> str:
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
    def extract_from_tool_content(
        turn_messages: list, auto_citations: list[dict], tools_used: list[dict]
    ) -> dict | None:
        """Fallback handler: use the raw tool response text as the answer when the
        agent stopped without generating a final AI message (e.g. model halted after
        receiving tool output without calling submit_final_answer).
        """
        for m in reversed(turn_messages):
            if getattr(m, "type", "") == "tool" and getattr(m, "name", "") == "search_document_knowledge":
                tool_text = m.content if isinstance(m.content, str) else str(m.content)
                # Strip METADATA blocks — keep only the readable result text
                cleaned = re.sub(r"METADATA:\s*\{.*?\}", "", tool_text, flags=re.DOTALL).strip()
                if cleaned:
                    return {"response": cleaned, "citations": auto_citations, "tools_used": tools_used}
        return None

    @staticmethod
    def extract_auto_citations(turn_messages: list) -> list[dict]:
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
    def parse_json_response(raw_text: str, auto_citations: list[dict], tools_used: list[dict]) -> dict | None:
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

    @classmethod
    async def extract_response(cls, messages: list) -> dict:
        """Extract structured response from agent messages."""
        turn_messages = cls.find_turn_messages(messages)
        tools_used = cls.collect_tools_used(turn_messages)

        # Handler 1: submit_final_answer tool call
        result = cls.extract_from_submit_tool(turn_messages, tools_used)
        if result is not None:
            if result.get("response") not in ("", "No answer found."):
                return result
            # Response is empty – fall through to raw-text extraction while
            # keeping citations from submit_final_answer.
            raw_text = cls.extract_raw_text(turn_messages)
            if raw_text:
                result["response"] = raw_text
                return result

        # Handler 2 & 3: raw text → JSON parse → plain text fallback
        raw_text = cls.extract_raw_text(turn_messages)
        auto_citations = cls.extract_auto_citations(turn_messages)

        if not raw_text:
            # Handler 2.5: tool content fallback
            result = cls.extract_from_tool_content(turn_messages, auto_citations, tools_used)
            if result is not None:
                return result
            return {"response": "No answer found.", "citations": [], "tools_used": tools_used}

        result = cls.parse_json_response(raw_text, auto_citations, tools_used)
        if result is not None:
            return result

        # Final fallback: plain text with auto-citations
        return {"response": raw_text, "citations": auto_citations, "tools_used": tools_used}

    @staticmethod
    def parse_thread_messages(messages: list) -> list[dict]:
        """Retrieve human/assistant messages for a thread, parsing any JSON content."""
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
