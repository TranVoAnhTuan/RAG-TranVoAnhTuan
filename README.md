# Agentic RAG PDF System

An intelligent PDF Q&A system utilizing RAG (Retrieval-Augmented Generation) combined with an Agentic architecture. This project allows users to upload PDF files, automatically extract content (including complex tables), store, index, and provide a smart AI chat interface to query information with high accuracy.

## Key Features

* **Smart Ingestion Pipeline (LangGraph):** An automated document processing workflow that includes: Extraction (Docling) -> Deduplication -> Cleaning -> Chunking -> Embedding & Loading.
* **Advanced Document Extraction:** Utilizes `Docling` to support OCR and accurately recognize table structures from PDFs.
* **Hybrid Search:** Combines semantic search (Dense Vector - SentenceTransformers) and keyword search (Sparse Vector - FastEmbed) using a Qdrant vector database.
* **Re-ranking:** Enhances search result relevance and precision using `jina-reranker-v3`.
* **Intelligent Agent:** Integrates LangChain and ChatOllama (`qwen3.5:2b`) with the ability to dynamically decide whether to call the document search tool or respond directly based on the conversation context.
* **VRAM Optimization:** Dynamic VRAM management. The system only loads the Dense Model and Reranker model into the GPU during inference, and releases them immediately afterward to save memory.
* **Intuitive UI:** A Streamlit-based frontend that allows users to upload files, track processing status, chat with the assistant, and visualize extracted table structures.
* **Multi-tier Storage Management:** MinIO (Object storage for PDFs), SQLite/MongoDB (Metadata storage), and Qdrant (Vector storage).

## 🛠 Tech Stack

* **Backend:** FastAPI, Python 3.11+
* **Frontend UI:** Streamlit
* **AI / LLM:** LangChain, LangGraph, Ollama (Qwen3.5:2b)
* **Embeddings & Reranker:** SentenceTransformers, FastEmbed, Jina AI
* **Databases:**
    * Vector DB: Qdrant
    * Object Storage: MinIO
    * Metadata DB: SQLite
* **Document Processing:** Docling, Langchain Text Splitters

## 📂 Project Structure

```text
.
├── app/
│   ├── agents/          # Agent logic (LangChain, Prompts, Tools)
│   ├── api/             # FastAPI Routes (/upload, /chat)
│   ├── core/            # Configs and Model Manager (Dynamic VRAM management)
│   ├── db/              # Database connections (MinIO, MongoDB, Qdrant, SQLite)
│   ├── services/        # RAG services (Hybrid search, reranking)
│   └── main.py          # FastAPI backend entry point
├── pipelines/           # Ingestion pipeline built with LangGraph
│   ├── check_duplicate.py # Document deduplication check
│   ├── chunk.py         # Text chunking (MarkdownHeaderTextSplitter)
│   ├── clean.py         # Text cleaning (Regex)
│   ├── embed.py         # Vector embedding and loading to Qdrant
│   ├── extract.py       # PDF extraction using Docling
│   ├── langgraph_ingestion.py # StateGraph workflow definition
│   └── state.py         # LangGraph State structure definition
├── ui/
│   └── app.py           # Streamlit user interface
└── .env                 # Environment variables