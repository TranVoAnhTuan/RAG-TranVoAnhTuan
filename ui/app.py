import streamlit as st
import requests

API_URL = "http://localhost:8000/api/v1"

st.set_page_config(page_title="PDF Agentic RAG", layout="wide")
st.title("🤖 Trợ Lý  Phân Tích PDF")

# Cột bên trái: Upload PDF
with st.sidebar:
    st.header("📂 Quản lý tài liệu")
    uploaded_file = st.file_uploader("Tải lên file PDF của bạn", type=["pdf"])
    
    if st.button("Xử lý và Lưu (ZenML)"):
        if uploaded_file is not None:
            with st.spinner("Đang xử lý PDF, chạy chunking & embedding... (Quá trình này có thể mất vài phút)"):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{API_URL}/upload", files=files)
                
                if response.status_code == 200:
                    st.success("✅ Thành công! Hệ thống đã học xong tài liệu.")
                else:
                    st.error(f"❌ Lỗi: {response.text}")
        else:
            st.warning("Vui lòng tải lên một file PDF trước.")

# Khung chat chính
st.header("💬 Trò chuyện với tài liệu")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Nhập câu hỏi
if prompt := st.chat_input("Hãy hỏi bất cứ điều gì về tài liệu..."):
    # Hiển thị câu hỏi của người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gọi API đến Agent
    with st.chat_message("assistant"):
        with st.spinner("Agent đang suy nghĩ và tìm kiếm dữ liệu..."):
            res = requests.post(f"{API_URL}/chat", json={"query": prompt})
            if res.status_code == 200:
                answer = res.json().get("answer", "Lỗi phản hồi từ server.")
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error("Có lỗi xảy ra khi kết nối tới Agent.")