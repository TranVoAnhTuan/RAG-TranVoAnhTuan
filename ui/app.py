import streamlit as st
import requests
import os
import uuid 
from streamlit_extras.bottom_container import bottom

API_URL = "http://127.0.0.1:8000/api/v1"
st.set_page_config(
    page_title="Document Intelligence Assistant", 
    layout="wide"
)

def initialize_session_state():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
        
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = "not_started"
    if "uploaded_files_info" not in st.session_state:
        st.session_state.uploaded_files_info = []
    # if "extracted_tables" not in st.session_state:
    #     st.session_state.extracted_tables = []

def render_sidebar():
    with st.sidebar:
        st.title("Setup")

        uploaded_file = st.file_uploader(
            "Upload Documents",
            type=["pdf"],
        )

        if uploaded_file:
            st.info(f"1 file uploaded")

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

                            st.session_state.extracted_tables = data.get("tables", []) 

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
            
            if message.get("citations"):
                with st.expander("📝 Citations"):
                    for i, cite in enumerate(message["citations"], 1):
                        h1 = cite.get('Header_1', 'Unknown Header')
                        h2 = cite.get('Header_2', '')
                        file_url = cite.get('file_url', '')
                        
                        header_text = f"**{i}. {h1}**"
                        if h2: 
                            header_text += f" - *{h2}*"
                            
                        st.markdown(header_text)
                        if file_url:
                            st.markdown(f"> 🔗 [View PDF]({file_url})")

    with bottom():
        prompt = st.chat_input("Ask a question about your documents or anything else...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching and synthesizing answer..."):
                try:
                    payload = {
                        "query": prompt,
                        "thread_id": st.session_state.thread_id
                    }
                    res = requests.post(f"{API_URL}/chat", json=payload)
                    
                    if res.status_code == 200:
                        raw_answer = res.json().get("answer", {})
                        
                        if isinstance(raw_answer, dict):
                            main_response = raw_answer.get("response", "No answer available.")
                            citations = raw_answer.get("citations", [])
                            
                            st.markdown(main_response)
                            if citations:
                                with st.expander("📝 Citations"):
                                    for i, cite in enumerate(citations, 1):
                                        h1 = cite.get('Header_1', 'Unknown Header')
                                        h2 = cite.get('Header_2', '')
                                        file_url = cite.get('file_url', '')
                                        
                                        header_text = f"**{i}. {h1}**"
                                        if h2: 
                                            header_text += f" - *{h2}*"
                                            
                                        st.markdown(header_text)
                                        if file_url:
                                            st.markdown(f"> 🔗 [View PDF]({file_url})")
                            
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": main_response,
                                "citations": citations
                            })
                        else:
                            st.markdown(str(raw_answer))
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": str(raw_answer)
                            })
                            
                    else:
                        st.error("Agent error")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to server")

def render_structure_viz():
    st.title("Document Structure")
    
    if st.session_state.processing_status != "completed":
        st.info("Please process documents first")
        return
        
    tables = st.session_state.extracted_tables
    
    if not tables:
        st.info("No tables found in document")
        return
        
    st.success(f"Extracted {len(tables)} tables")
    
    for tbl in tables:
        st.subheader(f"Table {tbl['table_number']}")
        st.dataframe(tbl['data'], use_container_width=True)
        st.divider()

def main():
    initialize_session_state()
    render_sidebar()

    # tab1, tab2 = st.tabs(["Chat", "Document Structure"])

    # with tab1:
    #     render_chat()

    # with tab2:
    #     render_structure_viz()
    st.title("Document Intelligence Assistant")
    render_chat()

if __name__ == "__main__":
    main()