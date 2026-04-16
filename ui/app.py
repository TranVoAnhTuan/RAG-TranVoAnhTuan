import streamlit as st
import requests
import uuid 
from streamlit_extras.bottom_container import bottom

API_URL = "http://127.0.0.1:8000/api/v1"
st.set_page_config(page_title="Document Intelligence Assistant", layout="wide")

def initialize_session_state():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = "not_started"
    if "uploaded_files_info" not in st.session_state:
        st.session_state.uploaded_files_info = []

def render_sidebar():
    with st.sidebar:
        st.title("Setup")

        uploaded_file = st.file_uploader("Upload Documents", type=["pdf"])

        if uploaded_file:
            st.info("1 file selected")

            if st.button("Process & Index", use_container_width=True):
                with st.spinner("Processing document..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    try:
                        response = requests.post(f"{API_URL}/upload", files=files)
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state.processing_status = "completed"

                            if uploaded_file.name not in st.session_state.uploaded_files_info:
                                st.session_state.uploaded_files_info.append(uploaded_file.name)

                            st.success("Documents indexed! You can now chat.")
                        else:
                            st.session_state.processing_status = "error"
                            st.error(f"Error: {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.session_state.processing_status = "error"
                        st.error("Cannot connect to FastAPI backend.")

        st.divider()
        
        if st.button("New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            if "pending_topics" in st.session_state:
                st.session_state.pending_topics = []
            st.rerun()

        st.caption(f"Current Thread ID: {st.session_state.thread_id}")
        
        st.divider()
        st.subheader("Status")

        if st.session_state.processing_status == "not_started":
            st.info("Ready to start")
        elif st.session_state.processing_status == "completed":
            st.success("Ready to chat")
        elif st.session_state.processing_status == "error":
            st.error("Error occurred")

        st.divider()
        st.subheader("Uploaded Files History")

        if st.session_state.uploaded_files_info:
            for i, fname in enumerate(st.session_state.uploaded_files_info, 1):
                st.write(f"{i}. {fname}")
        else:
            st.info("No files uploaded yet")

def render_chat():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message.get("needs_topic_clarification"):
                st.caption(f"Topic clarification requested for: {', '.join(message['needs_topic_clarification'])}")
            
            if message.get("citations"):
                with st.expander("📚 Citations"):
                    for i, cite in enumerate(message["citations"], 1):
                        h1 = cite.get('Header_1', 'Unknown Header')
                        h2 = cite.get('Header_2', '')
                        file_url = cite.get('file_url', '')
                        
                        header_text = f"**{i}. {h1}**" + (f" - *{h2}*" if h2 else "")
                        st.markdown(header_text)
                        if file_url:
                            st.markdown(f"> 🔗 [View PDF]({file_url})")

    prompt = None
    
    if st.session_state.get("pending_topics"):
        st.info("👇 Please select a topic to narrow down the search:")
        cols = st.columns(len(st.session_state.pending_topics))
        for idx, topic in enumerate(st.session_state.pending_topics):
            if cols[idx].button(topic, key=f"btn_{topic}"):
                # English hidden prompt for the agent
                prompt = f"I am asking specifically about the '{topic}' topic. Please answer my previous question based on this."
                st.session_state.pending_topics = []
                st.rerun()
                
    with bottom():
        user_input = st.chat_input("Ask a question about your documents or anything else...")
        if user_input:
            prompt = user_input
            if "pending_topics" in st.session_state:
                st.session_state.pending_topics = []

    if prompt:
        # Hide the system-generated follow-up prompt from the UI for a cleaner experience
        if not prompt.startswith("I am asking specifically about"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching and synthesizing answer..."):
                try:
                    payload = {"query": prompt, "thread_id": st.session_state.thread_id}
                    res = requests.post(f"{API_URL}/chat", json=payload)
                    
                    if res.status_code == 200:
                        raw_answer = res.json().get("answer", {})
                        
                        if isinstance(raw_answer, dict):
                            main_response = raw_answer.get("response", "No answer available.")
                            citations = raw_answer.get("citations", [])
                            clarification_topics = raw_answer.get("needs_topic_clarification", [])
                            
                            st.markdown(main_response)
                            
                            if clarification_topics:
                                st.session_state.pending_topics = clarification_topics
                                st.rerun()
                                
                            if citations:
                                with st.expander("📚 Citations"):
                                    for i, cite in enumerate(citations, 1):
                                        h1 = cite.get('Header_1', 'Unknown Header')
                                        h2 = cite.get('Header_2', '')
                                        file_url = cite.get('file_url', '')
                                        
                                        header_text = f"**{i}. {h1}**" + (f" - *{h2}*" if h2 else "")
                                        st.markdown(header_text)
                                        if file_url:
                                            st.markdown(f"> 🔗 [View PDF]({file_url})")
                            
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": main_response,
                                "citations": citations,
                                "needs_topic_clarification": clarification_topics
                            })
                        else:
                            st.markdown(str(raw_answer))
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": str(raw_answer)
                            })
                            
                    else:
                        st.error("Agent error occurred.")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to server.")

def main():
    initialize_session_state()
    render_sidebar()
    st.title("Document Intelligence Assistant")
    render_chat()

if __name__ == "__main__":
    main()