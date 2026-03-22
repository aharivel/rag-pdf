# rag-pdf

A local AI assistant that answers questions from your personal PDF library — running entirely on your own hardware, no cloud, no subscriptions.

## The idea

The goal was simple: repurpose a gaming PC as a local AI server and use it to query a collection of technical PDFs (books, research papers, documentation) without ever sending data to an external service.

The gaming PC handles the heavy lifting — running LLMs and embedding models via Ollama with GPU acceleration. The dev laptop handles the RAG pipeline — indexing PDFs into a local vector database and serving a query API that any OpenAI-compatible interface can connect to.

This is a learning project built while studying LLMs, RAG, and local AI — but it's also a genuinely useful tool. You can ask natural language questions across hundreds of pages of documentation and get grounded, sourced answers in seconds.

## How it works

```
[Dev laptop]                          [Gaming PC — GPU]
  PDFs on disk                          Ollama
  ChromaDB (vector store)    ←────→     ├─ LLM (qwen3:8b)
  LlamaIndex (RAG pipeline)             └─ Embedding model
  FastAPI (OpenAI-compatible API)            (nomic-embed-text-v2-moe)
  Open WebUI (or any compatible UI)
```

1. **Indexing** — PDFs are chunked, embedded via Ollama on the gaming PC, and stored in a local ChromaDB vector database
2. **Querying** — your question is embedded, the most relevant chunks are retrieved, and the LLM generates a grounded answer with source attribution

## Prerequisites

- [Ollama](https://ollama.com) running on a machine with a GPU, reachable on your local network with `OLLAMA_HOST=0.0.0.0`
- The following models pulled on that machine:
  ```
  ollama pull qwen3:8b
  ollama pull nomic-embed-text-v2-moe
  ```
- Python 3.11+

## Setup

```bash
git clone https://github.com/aharivel/rag-pdf
cd rag-pdf
make setup        # creates venv and installs dependencies
cp .env.example .env  # then edit .env with your Ollama machine IP
make check        # verify connection to Ollama
```

Edit `config.py` to select which PDF folders to index:

```python
PDF_FOLDERS = [
    f"{PDF_BASE}/AI",
    f"{PDF_BASE}/go",
    f"{PDF_BASE}/devops",
    # f"{PDF_BASE}/linux",   # uncomment to include
]
```

Then build the database and start the API:

```bash
make index    # embed and index your PDFs (run once, takes a while)
make start    # start the RAG API on port 8000
```

## Usage

```bash
make query Q="What is the difference between RAG and fine-tuning?"
make update-index   # add new PDFs without rebuilding from scratch
make clean          # wipe the database (needed after removing folders)
```

## UI

The RAG API exposes an OpenAI-compatible endpoint at `http://localhost:8000/v1`.
Any interface that supports OpenAI-style connections can use it — [Open WebUI](https://github.com/open-webui/open-webui) works particularly well.

## Tech stack

| Tool | Role |
|---|---|
| [Ollama](https://ollama.com) | Serves LLM and embedding models locally with GPU acceleration |
| [LlamaIndex](https://www.llamaindex.ai) | RAG pipeline — document loading, chunking, indexing, querying |
| [ChromaDB](https://www.trychroma.com) | Local persistent vector database |
| [FastAPI](https://fastapi.tiangolo.com) | OpenAI-compatible HTTP API layer |
| [qwen3:8b](https://ollama.com/library/qwen3) | Local LLM for answer generation |
| [nomic-embed-text-v2-moe](https://ollama.com/library/nomic-embed-text) | Embedding model for semantic search |

## Licence

MIT
