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
* **Clean Code:** Strictly adheres to SRP, DRY, and OOP principles (**Facade**, **Repository**, **Chain of Responsibility**). Fully typed and linted using `ruff`.

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
│   ├── agents/          # Agent orchestration (DemoAgent Facade)
│   ├── api/             # FastAPI Routes (/chat, /upload, /history, /chat/resume)
│   ├── core/            # App Configuration & ModelManager
│   ├── db/              # DB Singletons (MinIO, MongoDB, Qdrant, SQLite)
│   ├── utils/           # Shared logic (ResponseParser, ThreadRepo, RedisCache)
│   └── main.py          # FastAPI backend entry point
├── mcp_server/          # Standalone Model Context Protocol Server
├── pipelines/           # LangGraph PDF Ingestion Pipeline
├── ui/
│   └── app.py           # Streamlit Frontend
├── docker-compose.yml   # Multi-container deployment config
├── docker-compose.gpu.yml # GPU support overrides
├── pyproject.toml       # Poetry dependency management
└── .env                 # Environment variables
```

## 🔒 Corporate Proxy & SSL Certificates

If you are working behind a corporate proxy, you may need to provide your corporate CA certificates:
1. `cp combined-example.pem combined.pem`
2. Paste your Base64 encoded CA certificates into `combined.pem`.

---

## 📦 Running Options

### 1. Running with Docker WITH GPU (NVIDIA) 🏎️
Best performance. Requires **NVIDIA Container Toolkit** installed on your host.

```bash
# Build and run with GPU support
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

### 2. Running with Docker WITHOUT GPU (CPU Only) 🐌
Standard run. Models will run on CPU memory.

```bash
# Standard build and run
docker compose up -d --build
```

### 3. Local Development (No Docker) 💻
Run services individually. Requires Python 3.11+ and **Poetry**.

**Step 1: Install Dependencies**
```bash
poetry install
```

**Step 2: Start Local Databases**
You need to run the storage services. You can run them all via Docker:
```bash
docker compose up redis minio qdrant mongodb -d
```

**Step 3: Run Application Services**
Open 3 terminals:
- **Terminal 1 (Backend):** `poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`
- **Terminal 2 (MCP Server):** `poetry run python -m mcp_server.server`
- **Terminal 3 (Frontend):** `poetry run streamlit run ui/app.py`

---

## 🔗 Access URLs

- **Frontend UI:** `http://localhost:8501`
- **FastAPI Backend Docs:** `http://localhost:8000/docs`
- **MinIO Console:** `http://localhost:9001`
- **Qdrant Dashboard:** `http://localhost:6333/dashboard`

## 🛠 Development

**Linting and Formatting:**
```bash
poetry run ruff check .
poetry run ruff format .
```