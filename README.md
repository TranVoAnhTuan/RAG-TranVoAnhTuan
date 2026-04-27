# Agentic RAG System

An intelligent, multi-service Document Assistant built on an Agentic Retrieval-Augmented Generation (RAG) architecture. The system processes PDF documents, extracts and indexes their contents, and provides an interactive chat interface capable of both answering questions from internal knowledge and falling back to web search when necessary.

## 🚀 Key Features

* **Multi-Service Architecture:** Decoupled Microservices approach separating the Frontend (Streamlit), Backend (FastAPI), and Tools Server (FastMCP).
* **Agentic Orchestration:** Powered by a LangGraph agent that dynamically decides when to search internal documents, when to search the web (Tavily), or when to answer directly.
* **Model Context Protocol (MCP):** All agent tools (RAG retrieval, Web Search) and system prompts are hosted on an independent MCP server, enabling standardized plug-and-play AI integrations.
* **Smart Ingestion Pipeline (LangGraph):** Automated PDF processing workflow: Docling Extraction (OCR & Tables) -> Deduplication -> Cleaning -> Chunking -> Qdrant Embedding.
* **Hybrid Search & Reranking:** Fuses semantic search (Dense Vectors via SentenceTransformers) and keyword search (Sparse Vectors via FastEmbed) with RRF fusion, followed by Jina v3 cross-encoder reranking.
* **Human-in-the-Loop (HITL):** Web search execution (Tavily) requires explicit user approval via the Streamlit UI, preventing unintended external API calls.
* **High-Performance Caching:** Redis application-level caching ensures instant responses for duplicate queries.
* **VRAM Optimization:** Applies the **Template Method** design pattern to dynamically load models (Dense/Reranker) onto the GPU only during inference, instantly returning them to CPU to conserve VRAM.
* **Persistent Memory:** Uses MongoDB to store conversation checkpoints, allowing users to pause, resume, or replay past conversational threads.
* **Clean Code:** Strictly adheres to SRP, DRY, and OOP principles (Facade, Chain of Responsibility, Singleton). Fully typed and linted using `ruff`.

## 🛠 Tech Stack

* **Backend & API:** FastAPI, FastMCP (Model Context Protocol), Python 3.11+
* **Frontend UI:** Streamlit
* **AI / Orchestration:** LangChain, LangGraph, OpenAI (ChatOpenAI wrapper for local/remote LLMs)
* **Models:** gemma4:e4b (LLM), Jina Reranker v3, BM42 (Sparse), Qwen3-Embedding (Dense)
* **Databases:**
  * **Qdrant:** Vector database (Hybrid Search)
  * **MinIO:** Object storage (PDFs)
  * **MongoDB:** LangGraph checkpointing and conversational memory
  * **SQLite:** Document metadata
  * **Redis:** Application-level semantic caching
* **Tooling:** Poetry, Ruff, Docker Compose, Docling

## 📂 Project Structure

```text
.
├── app/
│   ├── agents/          # Agent orchestration (DemoAgent Facade, Chain of Responsibility)
│   ├── api/             # FastAPI Routes (/chat, /upload, /history, /chat/resume)
│   ├── core/            # App Configuration & ModelManager (GPU Template Method)
│   ├── db/              # DB Singletons (MinIO, MongoDB, Qdrant, SQLite)
│   └── main.py          # FastAPI backend entry point
├── mcp_server/          # Standalone Model Context Protocol Server
│   ├── prompts/         # Centralized System Prompts
│   ├── tools/           # MCP Tools (search_document_knowledge, tavily_search)
│   ├── config.py        # MCP Configuration
│   ├── model_manager.py # Isolated GPU manager for tools
│   ├── rag_service.py   # Hybrid retrieval & reranking logic
│   └── server.py        # FastMCP entry point
├── pipelines/           # LangGraph PDF Ingestion Pipeline
│   ├── check_duplicate.py
│   ├── chunk.py
│   ├── clean.py
│   ├── embed.py
│   ├── extract.py
│   ├── langgraph_ingestion.py
│   └── state.py
├── ui/
│   └── app.py           # Streamlit Frontend (Chat, History, HITL approvals)
├── docker-compose.yml   # Multi-container deployment config
├── pyproject.toml       # Poetry dependency management
├── ruff.toml            # Linter & Formatter configuration
└── .env                 # Environment variables
```

## 🐳 Running with Docker (Recommended)

The easiest way to run the entire stack (Frontend, Backend, MCP Server, Qdrant, MinIO, Redis, MongoDB) is using Docker Compose.

1. **Clone the repository.**
2. **Configure Environment:** Ensure your `.env` file is present in the root directory (configure LLM endpoints, MinIO keys, etc.).
3. **Build and Run:**
   ```bash
   docker compose up -d --build
   ```
4. **Access the Application:**
   - **Frontend UI:** `http://localhost:8501`
   - **FastAPI Backend Docs:** `http://localhost:8000/docs`
   - **MinIO Console:** `http://localhost:9001`

## 💻 Local Development (Poetry)

If you prefer to run services individually for development:

1. **Install Dependencies:**
   ```bash
   poetry install
   ```

2. **Lint and Format Code:**
   ```bash
   poetry run ruff check .
   poetry run ruff format .
   ```

3. **Run Services (Requires DBs to be running locally):**
   - **Backend:** `poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`
   - **MCP Server:** `poetry run python -m mcp_server.server`
   - **Frontend:** `poetry run streamlit run ui/app.py`