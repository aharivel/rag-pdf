"""
OpenAI-compatible FastAPI server that wraps your PDF RAG pipeline.
Open WebUI connects to this as if it were a standard LLM endpoint.

Usage:
  python rag_api.py
  → API available at http://localhost:8000
  → Add to Open WebUI as OpenAI connection: http://localhost:8000
"""
import time
import uuid

import chromadb
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

import config

app = FastAPI(title="PDF RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Index loaded once at startup
_index = None


def get_index():
    global _index
    if _index is None:
        print(f"Loading index from {config.CHROMA_PATH}...")
        Settings.embed_model = OllamaEmbedding(
            model_name=config.EMBED_MODEL,
            base_url=config.OLLAMA_EMBED_URL,
        )
        Settings.llm = Ollama(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_LLM_URL,
            request_timeout=120.0,
        )

        chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        try:
            chroma_collection = chroma_client.get_collection(config.CHROMA_COLLECTION)
        except Exception:
            raise RuntimeError(
                "ChromaDB collection not found. Run index_pdfs.py first."
            )

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        _index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
        print(f"Index loaded. {chroma_collection.count()} chunks available.")
    return _index


# ── OpenAI-compatible models ──────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "pdf-rag"
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "PDF RAG API"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ollama_llm": config.OLLAMA_LLM_URL,
        "ollama_embed": config.OLLAMA_EMBED_URL,
        "model": config.LLM_MODEL,
        "embed_model": config.EMBED_MODEL,
        "pdf_dir": config.PDF_DIR,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "pdf-rag",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
                "description": "RAG over your PDF knowledge base",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    # Extract the last user message
    user_message = next(
        (m.content for m in reversed(request.messages) if m.role == "user"),
        None,
    )
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")

    try:
        index = get_index()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    query_engine = index.as_query_engine(similarity_top_k=config.TOP_K)
    response = query_engine.query(user_message)

    # Build answer with source attribution
    sources = []
    for node in response.source_nodes:
        fname = node.metadata.get("file_name", "")
        score = round(node.score or 0, 3)
        if fname and fname not in [s["file"] for s in sources]:
            sources.append({"file": fname, "score": score})

    answer = str(response)
    if sources:
        source_lines = "\n".join(f"- {s['file']} (relevance: {s['score']})" for s in sources)
        answer += f"\n\n---\n**Sources from your PDF library:**\n{source_lines}"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "pdf-rag",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
