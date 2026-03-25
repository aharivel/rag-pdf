"""
OpenAI-compatible FastAPI server that wraps your PDF RAG pipeline.
Open WebUI connects to this as if it were a standard LLM endpoint.

Usage:
  python rag_api.py
  → API available at http://localhost:8000
  → Add to Open WebUI as OpenAI connection: http://localhost:8000
"""
import json
import time
import uuid

import chromadb
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

import config

SNIP_SYSTEM_PROMPT = (
    "When your response contains a command or code block worth saving as a "
    "reusable snippet, wrap it with:\n"
    "<snip category=\"CATEGORY\" headline=\"HEADLINE\" lang=\"LANG\">\n"
    "the code here\n"
    "</snip>\n"
    "Only tag concrete, reusable commands or code blocks — not prose "
    "explanations. Use short lowercase category names like: linux, bash, "
    "python, go, devops, git, docker, kubernetes. Omit lang if not applicable."
)

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
            context_window=config.get_num_ctx(config.LLM_MODEL),
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


def _build_chat_history(
    messages: list[dict],
) -> tuple[list[ChatMessage], str]:
    """Convert request messages to LlamaIndex ChatMessage history.

    Returns (chat_history, last_user_message) where chat_history
    contains all turns except the final user message.
    """
    if not messages:
        raise ValueError("messages list must not be empty")
    role_map = {
        "user": MessageRole.USER,
        "assistant": MessageRole.ASSISTANT,
    }
    chat_history = []
    for m in messages[:-1]:
        role = role_map.get(m["role"])
        if role is None:
            raise ValueError(f"Unknown message role: {m['role']!r}")
        chat_history.append(ChatMessage(role=role, content=m["content"]))
    last_user_msg = messages[-1]["content"]
    return chat_history, last_user_msg


# ── OpenAI-compatible models ──────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "pdf-rag"
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    llm_model: Optional[str] = None  # overrides config.LLM_MODEL if set
    chat_mode: Optional[str] = "condense_plus_context"  # condense_plus_context | context | simple


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


@app.get("/v1/ollama/models")
def list_ollama_models():
    """Return models available on the Ollama LLM server."""
    import httpx as _httpx
    try:
        resp = _httpx.get(f"{config.OLLAMA_LLM_URL}/api/tags", timeout=5.0)
        resp.raise_for_status()
        names = [m["name"] for m in resp.json().get("models", [])]
        return {"models": names}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Cannot reach Ollama: {e}")


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


@app.post("/v1/chat/completions/stream")
def chat_stream(request: ChatRequest):
    """Streaming SSE endpoint for the Textual chat app.

    SSE event contract (always ends with [DONE]):
    - Success: {"token": "..."} × N, {"sources": [...]}, [DONE]
    - Error:   {"token": "..."} × 0..N, {"error": "..."}, [DONE]
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    chat_history, last_user_msg = _build_chat_history(
        [{"role": m.role, "content": m.content} for m in request.messages]
    )

    active_model = request.llm_model or config.LLM_MODEL

    def generate():
        try:
            index = get_index()
        except RuntimeError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
            return

        llm_override = (
            Ollama(
                model=active_model,
                base_url=config.OLLAMA_LLM_URL,
                request_timeout=120.0,
                context_window=config.get_num_ctx(active_model),
            )
            if request.llm_model else None
        )

        mode = request.chat_mode or "condense_plus_context"
        chat_engine = index.as_chat_engine(
            chat_mode=mode,  # type: ignore[arg-type]
            streaming=True,
            similarity_top_k=config.TOP_K,
            system_prompt=SNIP_SYSTEM_PROMPT,
            **({"llm": llm_override} if llm_override else {}),
        )

        try:
            t_start = time.monotonic()
            t_first_token: float | None = None
            token_count = 0

            streaming_response = chat_engine.stream_chat(
                last_user_msg,
                chat_history=chat_history,
            )

            for token in streaming_response.response_gen:
                if t_first_token is None:
                    t_first_token = time.monotonic()
                token_count += 1
                yield f"data: {json.dumps({'token': token})}\n\n"

            t_end = time.monotonic()

            # Emit sources after all tokens
            sources = []
            seen = set()
            for node in streaming_response.source_nodes:
                fname = node.metadata.get("file_name", "")
                score = round(node.score or 0, 3)
                if fname and fname not in seen:
                    seen.add(fname)
                    sources.append({"file": fname, "score": score})
            yield f"data: {json.dumps({'sources': sources})}\n\n"

            # Emit generation stats for benchmarking / model comparison
            eval_s = (t_end - t_first_token) if t_first_token else 0.0
            total_s = t_end - t_start
            yield f"data: {json.dumps({'stats': {
                'model': active_model,
                'tokens': token_count,
                'tokens_per_sec': round(token_count / eval_s, 1) if eval_s > 0 else 0,
                'eval_s': round(eval_s, 2),
                'total_s': round(total_s, 2),
            }})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
