# How the PDF RAG system works

This document explains the code behind the system — what each file does,
why it's structured that way, and how the pieces connect.

---

## The big picture

A RAG system has two distinct phases that work very differently:

```
Phase 1 — INDEXING (run once, offline)
─────────────────────────────────────
Your PDFs → text → chunks → vectors → stored in ChromaDB

Phase 2 — QUERYING (run every time you ask a question)
───────────────────────────────────────────────────────
Your question → vector → search ChromaDB → top 5 chunks → LLM → answer
```

The key insight: **your PDFs are never sent to the LLM directly**.
They are converted into vectors (lists of numbers) that capture semantic
meaning. At query time, only the most relevant chunks are retrieved and
sent to the LLM as context.

---

## What is a vector / embedding?

An embedding model takes text and outputs a list of floating point numbers
(a vector). For example:

```
"DPDK uses kernel bypass for fast packet processing"
→ [0.023, -0.412, 0.887, 0.103, ..., -0.211]   # 768 numbers
```

The magic: **semantically similar text produces similar vectors**.
So "packet forwarding" and "network throughput" end up close together
in vector space, even though they share no words.

When you ask a question, it gets embedded the same way, then ChromaDB
finds the stored chunks whose vectors are closest to your question vector.
This is called a **similarity search** (cosine similarity).

---

## File by file

### `config.py` — Settings loader

```python
import os
from dotenv import load_dotenv

load_dotenv()  # reads your .env file into environment variables

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.1.111:11434")
LLM_MODEL       = os.getenv("LLM_MODEL", "qwen3:8b")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "nomic-embed-text-v2-moe")
...
```

**Why a separate config file?**
Every other script imports `config` instead of hardcoding values. This means
you change the model or IP in one place (`.env`) and it propagates everywhere.
It also keeps secrets (IPs, API keys) out of the code itself.

`load_dotenv()` is from the `python-dotenv` library. It reads your `.env` file
and injects each line as an environment variable, so `os.getenv()` can find them.

---

### `check_connection.py` — Preflight check

```python
r = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
models = [m["name"] for m in r.json().get("models", [])]
```

This just hits Ollama's REST API to list available models, then checks
that both `LLM_MODEL` and `EMBED_MODEL` are in the list.

**Ollama exposes a REST API** — it's not a Python library, it's an HTTP
server. Every interaction (embedding, inference) is an HTTP request.
LlamaIndex handles this for you under the hood, but it's good to know.

---

### `index_pdfs.py` — The indexing pipeline

This is the most important file to understand. Here's the flow:

```
1. Setup models (tell LlamaIndex which Ollama endpoints to use)
2. Discover all PDFs recursively
3. Load PDF pages into Document objects
4. Split Documents into smaller Nodes (chunks)
5. Embed each Node via Ollama → vector
6. Store vectors + text in ChromaDB
```

#### Step 1: Configure LlamaIndex settings

```python
Settings.embed_model = OllamaEmbedding(
    model_name=config.EMBED_MODEL,
    base_url=config.OLLAMA_BASE_URL,
)
Settings.llm = Ollama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL)
Settings.node_parser = SentenceSplitter(
    chunk_size=config.CHUNK_SIZE,    # 1024 tokens per chunk
    chunk_overlap=config.CHUNK_OVERLAP,  # 200 token overlap between chunks
)
```

`Settings` is a global config object in LlamaIndex. You set it once and
all subsequent operations inherit it. Think of it like a dependency injection
container — you wire up the models here and LlamaIndex uses them everywhere.

**Why chunk overlap?** If a concept spans a page boundary, the overlap
ensures neither chunk loses critical context. Imagine chunking a sentence
in half — with overlap, both chunks contain the full sentence.

#### Step 2: Load PDFs

```python
reader = SimpleDirectoryReader(
    input_files=[str(p) for p in all_pdfs],
    filename_as_id=True,
)
documents = reader.load_data(show_progress=True)
```

`SimpleDirectoryReader` uses `pypdf` under the hood to extract text from
each PDF page. Each page becomes a `Document` object with:
- `text`: the extracted text content
- `metadata`: dict with `file_name`, `file_path`, `page_label`, etc.

`filename_as_id=True` means the document ID is derived from the file path,
which is how the `--update` mode later identifies already-indexed files.

#### Step 3: Index into ChromaDB

```python
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True,
)
```

`VectorStoreIndex.from_documents()` does several things internally:
1. Runs `SentenceSplitter` on each Document → list of `Node` objects
2. For each Node, calls the embed model → gets a vector
3. Stores (vector + text + metadata) in ChromaDB

`StorageContext` tells LlamaIndex *where* to store things. By default it
would store in memory. We point it at ChromaDB so it persists to disk.

**ChromaDB** stores data in the `./chroma_db/` folder as SQLite + binary
files. Each entry has: an ID, the raw text, the vector, and the metadata dict.

---

### `query_rag.py` — CLI query tool

```
Your question
    │
    ▼
OllamaEmbedding.get_text_embedding(question)
    │  → sends POST to Ollama /api/embeddings
    │  → returns a vector (768 floats)
    ▼
ChromaDB.query(vector, n_results=TOP_K)
    │  → cosine similarity search
    │  → returns top 5 matching chunks + their metadata
    ▼
Build prompt:
    "Answer this question using the context below.
     Context: [chunk1] [chunk2] [chunk3] [chunk4] [chunk5]
     Question: your question"
    │
    ▼
Ollama LLM (qwen3:8b)
    │  → sends POST to Ollama /api/chat
    │  → streams response tokens
    ▼
Print answer + source filenames
```

The key part is the `as_query_engine()` call:

```python
query_engine = index.as_query_engine(
    similarity_top_k=config.TOP_K,  # retrieve 5 most relevant chunks
    streaming=False,
)
response = query_engine.query(question)
```

LlamaIndex wraps all of the above steps inside `query_engine.query()`.
It handles prompt construction, retrieval, and LLM call automatically.

The response object contains:
- `str(response)` → the LLM's answer text
- `response.source_nodes` → the list of chunks that were retrieved

---

### `rag_api.py` — The FastAPI server

This exposes your RAG pipeline as an **OpenAI-compatible HTTP API**.
Open WebUI (and any other tool that speaks OpenAI API) can talk to it.

#### Why OpenAI-compatible?

The OpenAI API has become a de facto standard. Open WebUI, Continue.dev,
and many other tools can connect to *any* server that speaks this protocol.
By implementing two endpoints, your local RAG looks like "just another model"
to any of these tools.

#### The two key endpoints

```python
@app.get("/v1/models")
def list_models():
    # Returns a list with one model: "pdf-rag"
    # Open WebUI calls this to populate the model dropdown
```

```python
@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    # 1. Extract the last user message from the conversation history
    # 2. Run it through the RAG query engine
    # 3. Format the response in OpenAI's JSON structure
    # 4. Append source filenames to the answer
```

The request/response format follows the OpenAI spec exactly:

```json
// Request (what Open WebUI sends)
{
  "model": "pdf-rag",
  "messages": [
    {"role": "user", "content": "How does DPDK work?"}
  ]
}

// Response (what we send back)
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "DPDK works by... \n\nSources: dpdk_guide.pdf"
    }
  }]
}
```

#### Global index caching

```python
_index = None

def get_index():
    global _index
    if _index is None:
        # load ChromaDB, setup models...
        _index = VectorStoreIndex.from_vector_store(...)
    return _index
```

The index is loaded once into memory when the first request arrives,
then reused for all subsequent requests. Loading ChromaDB on every
request would be very slow (~2-3 seconds per query).

---

## Data flow diagram (complete)

```
                    INDEXING
                    ────────
 PDF files
     │
     │  pypdf (text extraction)
     ▼
 Document objects (one per page)
     │
     │  SentenceSplitter (chunking, 1024 tokens, 200 overlap)
     ▼
 Node objects (chunks of text + metadata)
     │
     │  HTTP POST → Ollama (Windows PC)
     │  nomic-embed-text-v2-moe
     ▼
 Vectors (768 floats per chunk)
     │
     │  ChromaVectorStore
     ▼
 ChromaDB (./chroma_db/ on disk)


                    QUERYING
                    ────────
 User question
     │
     │  HTTP POST → Ollama (Windows PC)
     │  nomic-embed-text-v2-moe
     ▼
 Question vector (768 floats)
     │
     │  Cosine similarity search
     ▼
 Top 5 matching chunks (text + metadata)
     │
     │  Prompt construction:
     │  "Given this context: [chunks]
     │   Answer: [question]"
     │
     │  HTTP POST → Ollama (Windows PC)
     │  qwen3:8b
     ▼
 Answer text + source filenames
     │
     ▼
 CLI output / Open WebUI / API response
```

---

## Key libraries and what they do

| Library | Role |
|---|---|
| `llama-index-core` | Orchestrates the full RAG pipeline (loading, chunking, indexing, querying) |
| `llama-index-embeddings-ollama` | Adapter that sends embedding requests to Ollama's HTTP API |
| `llama-index-llms-ollama` | Adapter that sends LLM inference requests to Ollama's HTTP API |
| `llama-index-vector-stores-chroma` | Adapter that reads/writes vectors to ChromaDB |
| `chromadb` | Local vector database — stores and searches vectors efficiently |
| `pypdf` | Extracts text from PDF files (used internally by SimpleDirectoryReader) |
| `fastapi` | Web framework that serves the OpenAI-compatible HTTP API |
| `uvicorn` | ASGI server that runs the FastAPI app |
| `python-dotenv` | Loads `.env` file into environment variables |
| `rich` | Pretty terminal output (progress bars, coloured text, panels) |
| `httpx` | HTTP client used in `check_connection.py` to ping Ollama |

---

## Things to experiment with

Once you're comfortable with the basics, these are good levers to pull:

**`TOP_K` in `.env`** — how many chunks are retrieved per query.
Higher = more context for the LLM but slower and risks noise.
Try 3 (precise) vs 8 (broad).

**`CHUNK_SIZE`** in `config.py` — smaller chunks (512) give more precise
retrieval but lose context. Larger chunks (2048) give more context but
retrieval is less precise. Technical books often benefit from larger chunks.

**`CHUNK_OVERLAP`** in `config.py` — increase if you notice answers
getting cut off at chunk boundaries.

**Swap the LLM** — change `LLM_MODEL` in `.env` to `deepseek-r1:8b`
for harder reasoning questions, or `gemma3:latest` for faster responses.

**Try a different embedding model** — change `EMBED_MODEL` to
`qwen3-embedding:0.6b` and re-run `index_pdfs.py` to compare retrieval quality.
