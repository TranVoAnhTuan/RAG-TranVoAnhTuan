import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.tools import tool
from langchain.agents import create_agent




llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    temperature=0.7,
    max_new_tokens=1024,
)

chat_model = ChatHuggingFace(llm=llm)

@tool
def check_status(name: str) -> str:
    """Dùng để kiểm tra trạng thái của một người hoặc vật."""
    return f"{name} đang ở trạng thái: Sẵn sàng (Online)"

tools = [check_status]
system_prompt =  """
Bạn là AI assistant.

Bạn có 1 công cụ:
- check_status(name): kiểm tra trạng thái

QUY TẮC:
- Nếu người dùng hỏi về trạng thái (ví dụ: "A có online không?")
  → BẮT BUỘC dùng tool check_status
- Không tự trả lời trạng thái nếu chưa gọi tool
- Nếu câu hỏi không liên quan → trả lời bình thường

Trả lời ngắn gọn, rõ ràng.
"""
agent = create_agent(chat_model, tools=tools, system_prompt=system_prompt)

query = "A có online không?"

final_answer = ""

for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
):
    msg = step["messages"][-1]
    msg.pretty_print()

    if msg.type == "ai" and msg.content:
        final_answer = msg.content

print("\n=== FINAL ANSWER ===")
print(final_answer)