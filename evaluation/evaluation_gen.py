import asyncio
import json
import logging
import os

from app.agents.demo_agent import DemoAgent

# Configure local logging for the script
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def process_testset():
    input_file = "evaluation/testset_updated.json"
    output_file = "evaluation/testset_with_answers.json"

    logger.info("🚀 Initializing Agent...")
    agent = DemoAgent()
    await agent.connect_mcp()
    logger.info("✅ Agent connected to MCP server.")

    # 1. TÍNH NĂNG RESUME: Ưu tiên đọc file output nếu đã có (để tiếp tục phần đang làm dở)
    if os.path.exists(output_file):
        logger.info(f"📂 Tìm thấy file đang chạy dở '{output_file}'. Đọc dữ liệu để tiếp tục...")
        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)
    else:
        logger.info(f"📄 Bắt đầu mới từ file '{input_file}'...")
        with open(input_file, encoding="utf-8") as f:
            data = json.load(f)

    logger.info(f"Tổng số lượng: {len(data)} câu hỏi.\n")

    # 2. Vòng lặp xử lý
    for i, item in enumerate(data):
        if i % 10 == 0 and i != 0:
            logger.info("♻️ Restarting agent to free memory...")
            agent = DemoAgent()
            await agent.connect_mcp()
        # BỎ QUA nếu câu này đã được trả lời thành công trước đó
        # (Điều kiện: có key "answer", không phải lỗi, và không phải lỗi "Agent not initialized")
        if item.get("answer"):
            is_error = item["answer"].startswith("Error:")
            is_uninitialized = "Agent not initialized" in item["answer"]
            if not is_error and not is_uninitialized:
                logger.info(f"[{i + 1}/{len(data)}] Đã có câu trả lời, bỏ qua: {item.get('question')}")
                continue

        question = item.get("question")
        topic = item.get("topic", "General")
        if not question:
            continue

        logger.info(f"[{i + 1}/{len(data)}] Đang xử lý ({topic}): {question}")
        thread_id = f"eval_thread_{i}"

        try:
            # Chạy qua agent với topic tương ứng
            result = await agent.ask(question, thread_id=thread_id, topic=topic)

            # SỬA TẠI ĐÂY: Lấy trực tiếp từ result vì Agent không trả về key "answer" bọc ngoài
            item["answer"] = result.get("response", "No answer found.")
            item["model_citations"] = result.get("citations", [])

            logger.info(f"   -> Đã lấy được câu trả lời.")

        except Exception as e:
            logger.error(f"   -> ❌ Lỗi: {e}")
            item["answer"] = f"Error: {e!s}"
            item["model_citations"] = []

        # 3. LƯU NGAY VÀO FILE SAU MỖI CÂU
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"   -> 💾 Đã lưu tiến độ vào {output_file}.\n")

        await asyncio.sleep(1)

    logger.info(f"✅ Hoàn thành! Đã chạy qua toàn bộ danh sách.")


if __name__ == "__main__":
    asyncio.run(process_testset())
