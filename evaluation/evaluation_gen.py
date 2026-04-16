import json
import os
import asyncio
from app.agents.demo_agent import DemoAgent

async def process_testset():
    input_file = "/home/jacktran/RAG/experiment/rag_agentic_system/testset_updated.json"
    output_file = "testset_with_answers.json"

    print("🚀 Initializing Agent...")
    agent = DemoAgent()

    # 1. TÍNH NĂNG RESUME: Ưu tiên đọc file output nếu đã có (để tiếp tục phần đang làm dở)
    if os.path.exists(output_file):
        print(f"📂 Tìm thấy file đang chạy dở '{output_file}'. Đọc dữ liệu để tiếp tục...")
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        print(f"📄 Bắt đầu mới từ file '{input_file}'...")
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

    print(f"Tổng số lượng: {len(data)} câu hỏi.\n")

    # 2. Vòng lặp xử lý
    for i, item in enumerate(data):
        if i % 10 == 0 and i != 0:
            print("♻️ Restarting agent to free memory...")
            agent = DemoAgent()
        # BỎ QUA nếu câu này đã được trả lời thành công trước đó
        # (Điều kiện: có key "answer" và không phải là thông báo lỗi)
        if "answer" in item and item["answer"] and not item["answer"].startswith("Error:"):
            print(f"[{i+1}/{len(data)}] Đã có câu trả lời, bỏ qua: {item.get('question')}")
            continue

        question = item.get("question")
        if not question:
            continue

        print(f"[{i+1}/{len(data)}] Đang xử lý: {question}")
        thread_id = f"eval_thread_{i}"

        try:
            # Chạy qua agent
            result = await agent.ask(question, thread_id=thread_id)
            
            # SỬA TẠI ĐÂY: Lấy trực tiếp từ result vì Agent không trả về key "answer" bọc ngoài
            item["answer"] = result.get("response", "No answer found.")
            item["model_citations"] = result.get("citations", [])
            
            print(f"   -> Đã lấy được câu trả lời.")
            
        except Exception as e:
            print(f"   -> ❌ Lỗi: {e}")
            item["answer"] = f"Error: {str(e)}"
            item["model_citations"] = []
        
        # 3. LƯU NGAY VÀO FILE SAU MỖI CÂU
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"   -> 💾 Đã lưu tiến độ vào {output_file}.\n")

        await asyncio.sleep(1)

    print(f"✅ Hoàn thành! Đã chạy qua toàn bộ danh sách.")

if __name__ == "__main__":
    asyncio.run(process_testset())