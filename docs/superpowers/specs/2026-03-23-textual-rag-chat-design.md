# Textual RAG Chat CLI — Design Spec

**Date:** 2026-03-23
**Project:** `rag-pdf`
**Goal:** Build a Textual TUI chat app that talks to `rag_api.py` via streaming HTTP, supports true multi-turn conversation, and teaches both Textual internals and the RAG pipeline end-to-end.

---

## 1. Scope

### Files modified
- `rag_api.py` — add one streaming SSE endpoint; use a fresh `CondensePlusContextChatEngine` per request (stateless)

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
     │  async for token in rag_client.send_message(text)
     ▼
rag_client.py        (async HTTP client — owns history, parses SSE)
     │
     │  POST /v1/chat/completions/stream  (SSE)
     ▼
rag_api.py           (FastAPI — new streaming route, stateless per request)
     │
     │  fresh CondensePlusContextChatEngine per request
     ▼
ChromaDB + Ollama    (unchanged)
```

**Hard rules:**
- `chat_app.py` never imports `httpx`, `chromadb`, or `llama_index`.
- `rag_client.py` never imports `textual`.
- The server is fully stateless — no session, no shared engine state.

---

## 3. Data Flow

1. User types in the `Input` widget and presses Enter.
2. `on_input_submitted` adds a "You" bubble to the `VerticalScroll`, clears the input, creates an empty "Assistant" bubble, and calls `self.query(text)`.
3. `query` is a `@work` **async task** (no `thread=True`) — it runs on Textual's event loop, so widgets can be updated directly without `call_from_thread`.
4. Inside the worker, `async for token in rag_client.send_message(text)` yields:
   - Regular tokens (str) — accumulated and written to the assistant bubble via `bubble.update(accumulated)`.
   - A special final `{"sources": [...]}` token — appended below the answer text.
   - An error sentinel — rendered as an error-styled bubble instead.
5. On stream end, `rag_client` internally appends the full assistant reply to `conversation_history`.

**Why no `call_from_thread`:** Textual's `@work` (async, no `thread=True`) is an `asyncio.Task` running on the same event loop as the app. Widget mutations from async tasks are safe — `call_from_thread` is only needed when calling from an OS thread.

**Update batching:** To avoid thrashing the renderer at high token rates (50–100 tok/s), the worker accumulates tokens and calls `bubble.update()` at most every 50 ms using a simple `time.monotonic()` check, with a final flush when the stream ends. After each `bubble.update()` call, `await scroll.scroll_end(animate=False)` is called (the `await` is required — `scroll_end()` is a coroutine on `VerticalScroll`).

**Context window:** `conversation_history` is unbounded. Ollama's default context window (4096–8192 tokens) will eventually be exceeded in long conversations, causing silent truncation or LLM errors. A future enhancement should add a sliding window (e.g., keep only the last N turns), but this is out of scope for the initial implementation. The `/clear` command provides a manual escape hatch.

**Key learning moments:**
- **Textual:** `@work` async task, widget mutation from a task, `scroll_end()` after each update, CSS layout.
- **RAG:** how `CondensePlusContextChatEngine` condenses prior turns into a retrieval query before searching ChromaDB, then streams the LLM answer.
- **HTTP:** SSE wire format (`data: ...\n\n`), `httpx` async streaming, mid-stream error events.

---

## 4. `rag_api.py` Changes

### New route

```
POST /v1/chat/completions/stream
```

- Accepts the same `ChatRequest` body (`messages: List[Message]`).
- Extracts the last user message; builds `chat_history` as a list of LlamaIndex `ChatMessage` objects covering **all prior turns (both user and assistant)**, in order, excluding the final user message:
  ```python
  from llama_index.core.llms import ChatMessage, MessageRole
  role_map = {"user": MessageRole.USER, "assistant": MessageRole.ASSISTANT}
  chat_history = [
      ChatMessage(role=role_map[m.role], content=m.content)
      for m in request.messages[:-1]   # all turns except the last
  ]
  last_user_msg = request.messages[-1].content
  ```
- Creates a **fresh** `CondensePlusContextChatEngine` per request from the existing `_index` singleton:
  ```python
  chat_engine = get_index().as_chat_engine(
      chat_mode="condense_plus_context",
      streaming=True,
  )
  ```
  This is stateless: no shared mutable state, safe under any concurrency, no double-accumulation of history.
- The route is `def` (not `async def`) so FastAPI runs it in a threadpool — the blocking `get_index()` call is safe.
- Returns `StreamingResponse(generator(), media_type="text/event-stream")`.

### SSE event format and ordering contract

The SSE stream **always** ends with `data: [DONE]\n\n`, in both success and error cases. Events are ordered as follows:

**Success path:**
```
data: {"token": "..."}   ← zero or more, one per LLM token
data: {"token": "..."}
...
data: {"sources": [...]} ← exactly one; empty list [] if no source nodes
data: [DONE]
```

**Error path (mid-stream or pre-stream):**
```
data: {"token": "..."}   ← zero or more partial tokens already sent
data: {"error": "..."}   ← exactly one error event
data: [DONE]             ← always present; client uses this to stop iteration
```

**`{"sources": [...]}` notes:**
- Always emitted on the success path, even if `source_nodes` is empty (`[]`).
- Never emitted on the error path.
- Each entry: `{"file": "<filename>", "score": <float>}`.
- Deduplicated by filename (same logic as existing `/v1/chat/completions`).

Sources are extracted from `response.source_nodes` after `response_gen` is exhausted.

### Unchanged
The existing `/v1/chat/completions` endpoint is **not modified**. The MCP server, Open WebUI, and `query_rag.py` all continue to work.

---

## 5. `rag_client.py`

```python
class RagClient:
    base_url: str          # from env var CHAT_API_URL, default "http://localhost:8000"
    conversation_history: list[dict]  # {"role": "user"|"assistant", "content": str}

    async def send_message(self, text: str) -> AsyncIterator[str | dict]:
        # 1. append user message to history
        # 2. POST /v1/chat/completions/stream with full history
        # 3. parse SSE line by line:
        #    - skip "data: [DONE]"
        #    - JSON-decode each "data: ..." line
        #    - yield token str for {"token": ...}
        #    - yield dict for {"sources": ...}
        #    - raise / yield error sentinel for {"error": ...}
        # 4. on stream end: append accumulated assistant reply to history

    def clear_history(self) -> None:
        self.conversation_history.clear()
```

**httpx config:**
```python
httpx.AsyncClient(
    base_url=self.base_url,
    timeout=httpx.Timeout(None, connect=5.0),  # no read timeout (LLM is slow), fast connect fail
)
```

**SSE parsing:** iterate `response.aiter_lines()`, look for lines starting with `data: `, skip `[DONE]`, JSON-decode the rest.

**Base URL:** read from env var `CHAT_API_URL`; default `http://localhost:8000`. Does not import `config.py` (keeping the layer boundary clean).

---

## 6. `chat_app.py` — Textual UI

### Layout

```
┌─────────────────────────────────────────┐
│  PDF RAG Chat              [q] quit     │  ← Header
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
│    ║                               ║   │
│    ║ Sources: dpdk_guide.pdf 0.91  ║   │
│    ╚═══════════════════════════════╝   │
│                                  ↕ scroll
├─────────────────────────────────────────┤
│  > ░                                   │  ← Input (always focused)
└─────────────────────────────────────────┘
```

### Widgets

| Widget | Role | Textual concept learned |
|--------|------|------------------------|
| `App` | Root, CSS, keybindings | `compose()`, `BINDINGS`, CSS variables |
| `Header` | Title bar | Built-in widget |
| `VerticalScroll` | Scrollable history, auto-scrolls to bottom | Container, `scroll_end()` |
| `Static` | Each chat bubble; updated live during streaming | `update()`, Rich Markup |
| `Input` | Text entry at bottom, always focused | `on_input_submitted` event |
| `@work` | Background async SSE consumer per message | Async tasks, event loop safety |

### Special commands (typed in Input)
- `/clear` — calls `rag_client.clear_history()` and removes all bubbles from `VerticalScroll`
- `/quit` — exits the app (same as `q` binding)

### Entry point
```bash
python chat_app.py          # direct
make chat                   # Makefile target (to be added)
```

---

## 7. Error Handling

| Failure | Where caught | What user sees |
|---------|-------------|----------------|
| `rag_api.py` not running | `rag_client` catches `httpx.ConnectError` | Error bubble: "Cannot reach RAG API — is rag_api.py running?" |
| Non-200 HTTP response | `rag_client` checks `response.status_code` | Error bubble: "LLM unavailable (HTTP {status})" |
| Mid-stream LLM error | Server emits `{"error": "..."}` SSE event; client yields error sentinel | Error bubble with server message |
| Stream interrupted / truncated | `rag_client` catches `httpx.RemoteProtocolError` | Error bubble: "Stream interrupted — partial response shown" |
| httpx connect timeout | `rag_client` catches `httpx.ConnectTimeout` | Error bubble: "Connection timed out" |
| Empty input | `chat_app.py` early return in `on_input_submitted` | No action, Input stays focused |

All errors are yielded as a special `{"error": "..."}` dict from `rag_client` so `chat_app.py` renders them uniformly — no try/except in the UI layer.

---

## 8. Dependencies

**New (add to `requirements.txt`):**
- `textual` — TUI framework

**Already present:**
- `httpx` — already in `requirements.txt` (used by existing tooling)
- `fastapi` — `StreamingResponse` is built in, no new server-side dep

---

## 9. What You Will Learn

### Textual
- `App.compose()` — declarative widget tree
- CSS layout (Textual uses a subset of CSS for sizing/positioning)
- Event handlers (`on_input_submitted`, `on_key`)
- `@work` async task — runs on the event loop, direct widget mutation is safe
- `widget.update()` — live content mutation with Rich Markup
- `scroll_end()` — auto-scroll to latest message
- Update batching — accumulate tokens, render at ~20 fps

### RAG / LlamaIndex
- `CondensePlusContextChatEngine` — how prior turns are condensed into a retrieval query before hitting ChromaDB
- Streaming query engine — `response_gen` iterator over tokens
- Stateless multi-turn: fresh engine per request, full history passed each time — no server-side session needed

### HTTP / Async
- Server-Sent Events (SSE) wire format (`data: ...\n\n`, `[DONE]` sentinel)
- FastAPI `StreamingResponse` with a synchronous generator (runs in threadpool)
- `httpx.AsyncClient` with `aiter_lines()` for async SSE consumption
- Mid-stream error signalling (error event before `[DONE]`)
- `asyncio.Task` vs OS thread — why `call_from_thread` is not needed here
