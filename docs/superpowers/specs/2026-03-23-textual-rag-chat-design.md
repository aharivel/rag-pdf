# Textual RAG Chat CLI — Design Spec

**Date:** 2026-03-23
**Project:** `rag-pdf`
**Goal:** Build a Textual TUI chat app that talks to `rag_api.py` via streaming HTTP, supports true multi-turn conversation, and teaches both Textual internals and the RAG pipeline end-to-end.

---

## 1. Scope

### Files modified
- `rag_api.py` — add one streaming SSE endpoint; switch query engine to `CondensePlusContextChatEngine`

### Files created (inside `rag-pdf/`)
- `rag_client.py` — async HTTP client; owns conversation history; parses SSE stream
- `chat_app.py` — Textual TUI; pure UI layer; calls `rag_client`

### Files untouched
- `mcp_server.py`, `query_rag.py`, `config.py`, `index_pdfs.py`, `check_connection.py`

---

## 2. Architecture

Two clear layers with a hard boundary:

```
chat_app.py          (Textual — UI only)
     │
     │  yields str tokens / errors
     ▼
rag_client.py        (async HTTP client — RAG/network only)
     │
     │  POST /v1/chat/completions/stream  (SSE)
     ▼
rag_api.py           (FastAPI — existing, minimal addition)
     │
     │  LlamaIndex CondensePlusContextChatEngine
     ▼
ChromaDB + Ollama    (unchanged)
```

`chat_app.py` never imports `httpx`, `chromadb`, or `llama_index`.
`rag_client.py` never imports `textual`.

---

## 3. Data Flow

1. User types in the `Input` widget and presses Enter.
2. `on_input_submitted` adds a "You" bubble to the `VerticalScroll`, clears the input, spawns a Textual `@work` async worker.
3. Worker calls `await rag_client.send_message(text)` which:
   - Appends the message to the local `conversation_history` list.
   - POSTs `{ "messages": [...full history...] }` to `POST /v1/chat/completions/stream`.
   - Opens an `httpx.AsyncClient` streaming response.
   - Parses SSE lines (`data: {"token": "..."}`) and yields each token string.
4. Worker receives each token and calls `app.call_from_thread(assistant_bubble.update, accumulated_text)` to update the live bubble.
5. When the stream ends, `rag_client` appends the full assistant response to `conversation_history`.

**Key learning moments:**
- **Textual:** `@work` decorator, `call_from_thread` for thread-safe UI mutation, reactive `update()` on a `Static` widget.
- **RAG:** how `CondensePlusContextChatEngine` condenses prior turns into a retrieval query, then streams the LLM answer.
- **HTTP:** SSE wire format (`data: ...\n\n`), `httpx` async streaming.

---

## 4. `rag_api.py` Changes

Add a new route alongside the existing `/v1/chat/completions`:

```
POST /v1/chat/completions/stream
```

- Accepts the same `ChatRequest` body (`messages: List[Message]`).
- Builds `chat_history` (all messages except the last user turn) as LlamaIndex `ChatMessage` objects.
- Calls `chat_engine.stream_chat(last_user_msg, chat_history=chat_history)`.
- Returns `StreamingResponse` with `media_type="text/event-stream"`.
- Each SSE event: `data: {"token": "<tok>"}\n\n`
- Final event: `data: [DONE]\n\n`

The `chat_engine` is a module-level singleton (like `_index` today), initialised lazily on first request using `index.as_chat_engine(chat_mode="condense_plus_context", streaming=True)`.

The existing `/v1/chat/completions` endpoint is **not modified** — MCP and Open WebUI continue to work.

---

## 5. `rag_client.py`

```
RagClient
  conversation_history: list[dict]   # {"role": "user"|"assistant", "content": str}
  base_url: str                       # default http://localhost:8000

  async send_message(text: str) -> AsyncIterator[str]
    - appends user message to history
    - POSTs to /v1/chat/completions/stream with full history
    - parses SSE, yields token strings
    - on stream end: appends accumulated assistant reply to history

  clear_history()                     # /clear command support
```

Uses `httpx.AsyncClient` with `stream=True`. SSE parsing: split on `\n`, look for lines starting with `data: `, skip `[DONE]`, JSON-decode the rest.

---

## 6. `chat_app.py` — Textual UI

### Layout

```
┌─────────────────────────────────────────┐
│  PDF RAG Chat              [q] quit     │
├─────────────────────────────────────────┤
│  VerticalScroll (message history)       │
│                                         │
│    You                                  │
│    ╔═══════════════════════════════╗   │
│    ║ How does DPDK handle packets? ║   │
│    ╚═══════════════════════════════╝   │
│                                         │
│    Assistant                            │
│    ╔═══════════════════════════════╗   │
│    ║ DPDK uses poll-mode drivers…  ║   │
│    ║ Sources: dpdk_guide.pdf 0.91  ║   │
│    ╚═══════════════════════════════╝   │
│                                  ↕ scroll
├─────────────────────────────────────────┤
│  > ░                                   │
└─────────────────────────────────────────┘
```

### Widgets

| Widget | Role | Textual concept learned |
|--------|------|------------------------|
| `App` | Root, CSS, keybindings (`q` to quit) | `compose()`, `BINDINGS`, CSS |
| `VerticalScroll` | Scrollable history, auto-scrolls to bottom | Container, `scroll_end()` |
| `Static` | Each chat bubble; updated live during streaming | `update()`, Markup/Markdown |
| `Input` | Text entry at bottom, always focused | `on_input_submitted` event |
| `@work` | Background SSE consumer per message | Async workers, `call_from_thread` |

### Special commands (typed in the input)
- `/clear` — clears visual history and `rag_client.conversation_history`
- `/quit` — exits the app

---

## 7. Error Handling

| Failure | Where caught | What user sees |
|---------|-------------|----------------|
| `rag_api.py` not running | `rag_client` catches `httpx.ConnectError` | Error bubble: "Cannot reach RAG API — is rag_api.py running?" |
| Ollama / LLM unreachable | `rag_api.py` returns non-200; `rag_client` detects | Error bubble: "LLM unavailable — is Ollama running?" |
| Empty input | `chat_app.py` early return in handler | No action, input stays focused |
| Stream interrupted | `rag_client` catches `httpx.RemoteProtocolError` | Error bubble: "Stream interrupted — partial response shown" |

All errors are yielded as a special sentinel from `rag_client` so `chat_app.py` renders them uniformly without any try/except in the UI layer.

---

## 8. Dependencies

New packages needed (add to `requirements.txt`):
- `textual` — TUI framework
- `httpx` — async HTTP client with streaming support

`rag_api.py` already uses `fastapi` which supports `StreamingResponse` natively — no new server-side dependency.

---

## 9. What You Will Learn

### Textual
- `App.compose()` — declarative widget tree
- CSS layout (Textual uses a subset of CSS for sizing/positioning)
- Event handlers (`on_input_submitted`, `on_key`)
- `@work` async worker decorator
- `call_from_thread` — the key primitive for streaming UI updates
- `widget.update()` — live content mutation

### RAG / LlamaIndex
- `CondensePlusContextChatEngine` — how prior turns are condensed into a retrieval query
- Streaming query engine (`response_gen` iterator)
- Stateless multi-turn: history passed per-request, no server session needed

### HTTP / Async
- Server-Sent Events (SSE) wire format
- FastAPI `StreamingResponse` with a generator
- `httpx.AsyncClient` streaming context manager
- SSE parsing (line-by-line, `data:` prefix, `[DONE]` sentinel)
