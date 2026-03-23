# Textual RAG Chat CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a streaming Textual TUI chat app to `rag-pdf` that talks to `rag_api.py` via SSE, with true multi-turn conversation history.

**Architecture:** Two new files (`rag_client.py` for HTTP/SSE logic, `chat_app.py` for Textual UI) plus one new endpoint in `rag_api.py`. The server is stateless — a fresh `CondensePlusContextChatEngine` is created per request, with full history passed in each time. The Textual app uses a `@work` async task to consume the SSE stream and update a `Static` bubble live.

**Tech Stack:** Python 3.13, FastAPI + `StreamingResponse`, LlamaIndex `CondensePlusContextChatEngine`, `httpx` async streaming, Textual TUI, pytest + pytest-asyncio.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `requirements.txt` | Modify | Add `textual`, `pytest-asyncio` |
| `rag_api.py` | Modify | Add `POST /v1/chat/completions/stream` SSE endpoint |
| `rag_client.py` | Create | Async HTTP client; SSE parsing; conversation history |
| `chat_app.py` | Create | Textual TUI; pure UI; calls `rag_client` |
| `Makefile` | Modify | Add `make chat` and `make test` targets |
| `tests/__init__.py` | Create | Empty; marks tests/ as a package |
| `tests/test_rag_api_helpers.py` | Create | Unit tests for `_build_chat_history` |
| `tests/test_rag_client.py` | Create | Unit tests for SSE parsing and `RagClient` |
| `tests/test_chat_app.py` | Create | Smoke tests for Textual app composition |

---

## Task 1: Install Dependencies and Create Test Scaffolding

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`

- [ ] **Step 1: Add new dependencies to requirements.txt**

Open `requirements.txt` and append these two lines at the end:
```
textual>=0.70.0
pytest-asyncio>=0.23.0
```

- [ ] **Step 2: Create `pytest.ini` to configure asyncio mode**

Create `pytest.ini` in the project root:
```ini
[pytest]
asyncio_mode = auto
```

This prevents `pytest-asyncio >= 0.21` from silently skipping async tests in strict mode. With `asyncio_mode = auto`, all async test functions run automatically.

- [ ] **Step 3: Install the new dependencies**

```bash
cd /home/aharivel/Documents/rag-pdf
.venv/bin/pip install textual pytest-asyncio
```

Expected: both packages install without errors.

- [ ] **Step 4: Verify textual is importable**

```bash
.venv/bin/python -c "import textual; print(textual.__version__)"
```

Expected: prints a version number like `0.70.0` or higher.

- [ ] **Step 4: Create the tests package**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 5: Verify pytest finds the tests directory**

```bash
.venv/bin/python -m pytest tests/ --collect-only
```

Expected: `no tests ran` (empty test suite, but no errors).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pytest.ini tests/__init__.py
git commit -m "chore: add textual and pytest-asyncio, create tests scaffolding"
```

---

## Task 2: Add `_build_chat_history` Helper to `rag_api.py` (TDD)

This is a pure helper function — no Ollama or ChromaDB needed to test it.

**Files:**
- Create: `tests/test_rag_api_helpers.py`
- Modify: `rag_api.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rag_api_helpers.py`:

```python
"""Tests for rag_api.py helper functions."""
import pytest
from llama_index.core.llms import MessageRole


def test_build_chat_history_empty_messages():
    """Single user message → empty history, message extracted."""
    from rag_api import _build_chat_history
    messages = [{"role": "user", "content": "Hello"}]
    history, last = _build_chat_history(messages)
    assert history == []
    assert last == "Hello"


def test_build_chat_history_one_exchange():
    """One user + one assistant turn before the new user message."""
    from rag_api import _build_chat_history
    messages = [
        {"role": "user", "content": "What is DPDK?"},
        {"role": "assistant", "content": "DPDK is a data plane kit."},
        {"role": "user", "content": "How does it handle packets?"},
    ]
    history, last = _build_chat_history(messages)
    assert len(history) == 2
    assert history[0].role == MessageRole.USER
    assert history[0].content == "What is DPDK?"
    assert history[1].role == MessageRole.ASSISTANT
    assert history[1].content == "DPDK is a data plane kit."
    assert last == "How does it handle packets?"


def test_build_chat_history_preserves_order():
    """History items appear in the same order as the input messages."""
    from rag_api import _build_chat_history
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
        {"role": "user", "content": "Q3"},
    ]
    history, last = _build_chat_history(messages)
    assert len(history) == 4
    assert [m.content for m in history] == ["Q1", "A1", "Q2", "A2"]
    assert last == "Q3"
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
.venv/bin/python -m pytest tests/test_rag_api_helpers.py -v
```

Expected: `ImportError: cannot import name '_build_chat_history' from 'rag_api'`

- [ ] **Step 3: Add the helper function to `rag_api.py`**

Add this import near the top of `rag_api.py` (after the existing llama_index imports):

```python
from llama_index.core.llms import ChatMessage, MessageRole
```

Then add this function after the `get_index()` function (before the route definitions):

```python
def _build_chat_history(
    messages: list[dict],
) -> tuple[list[ChatMessage], str]:
    """Convert request messages to LlamaIndex ChatMessage history.

    Returns (chat_history, last_user_message) where chat_history
    contains all turns except the final user message.
    """
    role_map = {
        "user": MessageRole.USER,
        "assistant": MessageRole.ASSISTANT,
    }
    chat_history = [
        ChatMessage(role=role_map[m["role"]], content=m["content"])
        for m in messages[:-1]
    ]
    last_user_msg = messages[-1]["content"]
    return chat_history, last_user_msg
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
.venv/bin/python -m pytest tests/test_rag_api_helpers.py -v
```

Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add rag_api.py tests/test_rag_api_helpers.py
git commit -m "feat: add _build_chat_history helper to rag_api"
```

---

## Task 3: Add Streaming SSE Endpoint to `rag_api.py`

**Files:**
- Modify: `rag_api.py`

- [ ] **Step 1: Add the streaming route**

Add these imports at the top of `rag_api.py` (after existing imports):

```python
import json
from fastapi.responses import StreamingResponse
```

Then add this route after the existing `/v1/chat/completions` route:

```python
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

    def generate():
        try:
            index = get_index()
        except RuntimeError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
            return

        chat_engine = index.as_chat_engine(
            chat_mode="condense_plus_context",
            streaming=True,
        )

        try:
            streaming_response = chat_engine.stream_chat(
                last_user_msg,
                chat_history=chat_history,
            )

            for token in streaming_response.response_gen:
                yield f"data: {json.dumps({'token': token})}\n\n"

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

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

- [ ] **Step 2: Start the API server and test the endpoint with curl**

> **Note:** This step requires the GPU machine (Windows PC at `192.168.1.111`) to be on and Ollama running. If it's offline, you'll get `data: {"error": "..."}` instead of tokens — that's the error path working correctly, not a bug.

In one terminal, start the server:
```bash
.venv/bin/python rag_api.py
```

In another terminal, send a test request:
```bash
curl -N -X POST http://localhost:8000/v1/chat/completions/stream \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is DPDK?"}]}'
```

Expected: a stream of SSE lines like:
```
data: {"token": "DPDK"}
data: {"token": " stands"}
...
data: {"sources": [{"file": "...", "score": 0.91}]}
data: [DONE]
```

- [ ] **Step 3: Verify the existing endpoint still works**

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is DPDK?"}]}' | python3 -m json.tool
```

Expected: standard JSON response (not SSE), same as before.

- [ ] **Step 4: Commit**

```bash
git add rag_api.py
git commit -m "feat: add /v1/chat/completions/stream SSE endpoint"
```

---

## Task 4: Create `rag_client.py` (TDD)

**Files:**
- Create: `rag_client.py`
- Create: `tests/test_rag_client.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rag_client.py`:

```python
"""Tests for rag_client.py."""
import json
import pytest
import pytest_asyncio


# ── SSE parser ────────────────────────────────────────────────────────────────

def test_parse_sse_line_token():
    from rag_client import _parse_sse_line
    result = _parse_sse_line('data: {"token": "hello"}')
    assert result == {"token": "hello"}


def test_parse_sse_line_sources():
    from rag_client import _parse_sse_line
    result = _parse_sse_line('data: {"sources": [{"file": "foo.pdf", "score": 0.9}]}')
    assert result == {"sources": [{"file": "foo.pdf", "score": 0.9}]}


def test_parse_sse_line_error():
    from rag_client import _parse_sse_line
    result = _parse_sse_line('data: {"error": "LLM unavailable"}')
    assert result == {"error": "LLM unavailable"}


def test_parse_sse_line_done():
    from rag_client import _parse_sse_line
    result = _parse_sse_line("data: [DONE]")
    assert result is None  # signals end of stream


def test_parse_sse_line_empty():
    from rag_client import _parse_sse_line
    assert _parse_sse_line("") is None
    assert _parse_sse_line(":keep-alive") is None


# ── History management ────────────────────────────────────────────────────────

def test_clear_history():
    from rag_client import RagClient
    client = RagClient()
    client.conversation_history.append({"role": "user", "content": "hi"})
    client.clear_history()
    assert client.conversation_history == []


# ── send_message (mocked HTTP) ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_send_message_streams_tokens():
    """send_message yields token strings then a sources dict."""
    from rag_client import RagClient
    import httpx
    from unittest.mock import AsyncMock, MagicMock, patch

    sse_lines = [
        'data: {"token": "Hello"}',
        'data: {"token": " world"}',
        'data: {"sources": [{"file": "a.pdf", "score": 0.9}]}',
        "data: [DONE]",
        "",
    ]

    async def fake_aiter_lines():
        for line in sse_lines:
            yield line

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.aiter_lines = fake_aiter_lines

    async def fake_stream(*args, **kwargs):
        # Simulate httpx AsyncClient.stream() context manager
        class FakeStream:
            async def __aenter__(self_inner):
                return mock_response
            async def __aexit__(self_inner, *a):
                pass
        return FakeStream()

    client = RagClient(base_url="http://fake:9999")

    with patch.object(client._http, "stream", side_effect=fake_stream):
        results = []
        async for item in client.send_message("hi"):
            results.append(item)

    assert results[0] == "Hello"
    assert results[1] == " world"
    assert results[2] == {"sources": [{"file": "a.pdf", "score": 0.9}]}
    # History updated
    assert client.conversation_history[-2] == {"role": "user", "content": "hi"}
    assert client.conversation_history[-1] == {
        "role": "assistant", "content": "Hello world"
    }


@pytest.mark.asyncio
async def test_send_message_connect_error():
    """ConnectError yields an error dict and does not crash."""
    from rag_client import RagClient
    import httpx
    from unittest.mock import patch

    client = RagClient(base_url="http://fake:9999")

    async def raise_connect(*a, **kw):
        raise httpx.ConnectError("refused")

    class FakeStream:
        async def __aenter__(self):
            raise httpx.ConnectError("refused")
        async def __aexit__(self, *a):
            pass

    with patch.object(client._http, "stream", return_value=FakeStream()):
        results = []
        async for item in client.send_message("hi"):
            results.append(item)

    assert len(results) == 1
    assert "error" in results[0]
    assert "Cannot reach RAG API" in results[0]["error"]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/bin/python -m pytest tests/test_rag_client.py -v
```

Expected: `ModuleNotFoundError: No module named 'rag_client'`

- [ ] **Step 3: Implement `rag_client.py`**

Create `rag_client.py` in the project root:

```python
"""Async HTTP client for the PDF RAG streaming API.

Owns conversation history and SSE stream parsing.
Does not import anything from textual.
"""
from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

import httpx


def _parse_sse_line(line: str) -> dict | None:
    """Parse a single SSE line.

    Returns:
        dict  — parsed JSON payload (token, sources, or error event)
        None  — line should be ignored ([DONE], empty, comment)
    """
    if not line.startswith("data: "):
        return None
    payload = line[len("data: "):]
    if payload == "[DONE]":
        return None
    return json.loads(payload)


class RagClient:
    """Async client for POST /v1/chat/completions/stream.

    Maintains conversation history across calls.
    All network errors are yielded as {"error": "..."} dicts
    so callers never need to catch exceptions.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or os.getenv(
            "CHAT_API_URL", "http://localhost:8000"
        )
        self.conversation_history: list[dict] = []
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(None, connect=5.0),
        )

    def clear_history(self) -> None:
        """Reset conversation history (used by /clear command)."""
        self.conversation_history.clear()

    async def send_message(self, text: str) -> AsyncIterator[str | dict]:
        """Send a message and stream the response.

        Yields:
            str   — a token from the LLM answer
            dict  — {"sources": [...]} once after all tokens
            dict  — {"error": "..."} if something went wrong
        """
        self.conversation_history.append({"role": "user", "content": text})
        accumulated = ""

        try:
            async with self._http.stream(
                "POST",
                "/v1/chat/completions/stream",
                json={"messages": self.conversation_history},
            ) as response:
                if response.status_code != 200:
                    yield {"error": f"LLM unavailable (HTTP {response.status_code})"}
                    self.conversation_history.pop()  # roll back user message
                    return

                async for line in response.aiter_lines():
                    parsed = _parse_sse_line(line)
                    if parsed is None:
                        continue
                    if "token" in parsed:
                        accumulated += parsed["token"]
                        yield parsed["token"]
                    elif "sources" in parsed:
                        yield parsed  # {"sources": [...]}
                    elif "error" in parsed:
                        yield parsed  # {"error": "..."}
                        break

        except httpx.ConnectError:
            yield {"error": "Cannot reach RAG API — is rag_api.py running?"}
            self.conversation_history.pop()
            return
        except httpx.ConnectTimeout:
            yield {"error": "Connection timed out — is rag_api.py running?"}
            self.conversation_history.pop()
            return
        except httpx.RemoteProtocolError:
            yield {"error": "Stream interrupted — partial response shown"}
            # Keep partial accumulated text in history

        if accumulated:
            self.conversation_history.append(
                {"role": "assistant", "content": accumulated}
            )
```

- [ ] **Step 4: Run the tests**

```bash
.venv/bin/python -m pytest tests/test_rag_client.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add rag_client.py tests/test_rag_client.py
git commit -m "feat: add rag_client with SSE parsing and conversation history"
```

---

## Task 5: Create `chat_app.py` — Layout and Skeleton

**Files:**
- Create: `chat_app.py`
- Create: `tests/test_chat_app.py`

- [ ] **Step 1: Write a smoke test for app composition**

Create `tests/test_chat_app.py`:

```python
"""Smoke tests for the Textual chat app."""
import pytest
from unittest.mock import MagicMock, AsyncMock


@pytest.mark.asyncio
async def test_app_composes_without_error():
    """App starts and has the expected widgets."""
    from chat_app import RagChatApp
    app = RagChatApp()
    async with app.run_test() as pilot:
        # Header should be present
        assert app.query_one("Header")
        # VerticalScroll should be present
        assert app.query_one("#history")
        # Input should be present and focused
        input_widget = app.query_one("Input")
        assert input_widget is not None


@pytest.mark.asyncio
async def test_quit_binding_exits_app():
    """Pressing q exits the app."""
    from chat_app import RagChatApp
    app = RagChatApp()
    async with app.run_test() as pilot:
        await pilot.press("q")
    # If we reach here without hanging, the app exited cleanly
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/test_chat_app.py -v
```

Expected: `ModuleNotFoundError: No module named 'chat_app'`

- [ ] **Step 3: Implement the app skeleton**

Create `chat_app.py`:

```python
"""Textual TUI chat app for the PDF RAG pipeline.

Usage:
    python chat_app.py
    make chat

Architecture:
    This file is pure UI. All network and RAG logic lives in rag_client.py.
    Data flows: Input → @work async task → rag_client.send_message() → Static.update()
"""
from __future__ import annotations

import time

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Header, Input, Static
from textual.containers import VerticalScroll

from rag_client import RagClient


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
Screen {
    background: $surface;
}

#history {
    height: 1fr;          /* takes all space above the input */
    padding: 0 1;
}

.bubble-user {
    background: $primary-darken-3;
    color: $text;
    border: round $primary;
    margin: 1 0;
    padding: 0 1;
}

.bubble-assistant {
    background: $surface-darken-1;
    color: $text;
    border: round $accent;
    margin: 1 0;
    padding: 0 1;
}

.bubble-error {
    background: $error-darken-3;
    color: $text;
    border: round $error;
    margin: 1 0;
    padding: 0 1;
}

.bubble-label {
    color: $text-muted;
    text-style: bold;
    margin: 1 0 0 0;
}

Input {
    dock: bottom;
    margin: 0 1 1 1;
}
"""


# ── App ───────────────────────────────────────────────────────────────────────

class RagChatApp(App):
    """PDF RAG Chat — a Textual TUI for your local knowledge base.

    Key Textual concepts demonstrated here:
    - compose(): declarative widget tree
    - CSS: layout and theming
    - BINDINGS: keyboard shortcuts
    - on_input_submitted: event handler
    - @work: async task for background streaming
    - widget.update(): live content mutation
    - scroll_end(): auto-scroll to bottom
    """

    TITLE = "PDF RAG Chat"
    CSS = CSS

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._client = RagClient()

    def compose(self) -> ComposeResult:
        """Declare the widget tree.

        Textual calls this once at startup. yield each widget in
        the order they should appear top-to-bottom.
        """
        yield Header()
        yield VerticalScroll(id="history")
        yield Input(placeholder="Ask your PDF library… (/clear to reset, /quit to exit)")

    def on_mount(self) -> None:
        """Called after compose(). Focus the input immediately."""
        self.query_one(Input).focus()

    # ── Input handling ────────────────────────────────────────────────────────

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter in the Input widget.

        This is a Textual event handler — the method name encodes
        the widget type and event: on_<WidgetType>_<event_name>.
        """
        text = event.value.strip()
        event.input.clear()

        if not text:
            return

        # Special commands
        if text == "/clear":
            await self._do_clear()
            return
        if text == "/quit":
            self.exit()
            return

        await self._submit_query(text)

    async def _submit_query(self, text: str) -> None:
        """Add bubbles and start the streaming worker."""
        history = self.query_one("#history", VerticalScroll)

        # User bubble
        await history.mount(Static(f"[bold]You[/bold]", classes="bubble-label"))
        await history.mount(Static(text, classes="bubble-user"))

        # Empty assistant bubble — will be filled by the worker
        await history.mount(Static("[bold]Assistant[/bold]", classes="bubble-label"))
        assistant_bubble = Static("", classes="bubble-assistant")
        await history.mount(assistant_bubble)

        await history.scroll_end(animate=False)

        # Launch the streaming worker (async task, runs on the event loop)
        self._stream(text, assistant_bubble)

    async def _do_clear(self) -> None:
        """Clear visual history and conversation history."""
        self._client.clear_history()
        history = self.query_one("#history", VerticalScroll)
        await history.remove_children()

    # ── Streaming worker ──────────────────────────────────────────────────────

    @work
    async def _stream(self, text: str, bubble: Static) -> None:
        """Consume the SSE stream and update the assistant bubble live.

        This is a Textual @work async task — it runs on the same event loop
        as the app, so we can call widget.update() directly.

        Update batching: we accumulate tokens and re-render at most every
        50ms to avoid thrashing the renderer at high token rates.
        """
        history = self.query_one("#history", VerticalScroll)
        accumulated = ""
        last_render = time.monotonic()
        RENDER_INTERVAL = 0.05  # 50ms → ~20 fps

        async for item in self._client.send_message(text):
            if isinstance(item, str):
                # LLM token
                accumulated += item
                now = time.monotonic()
                if now - last_render >= RENDER_INTERVAL:
                    bubble.update(accumulated)
                    await history.scroll_end(animate=False)
                    last_render = now

            elif isinstance(item, dict) and "sources" in item:
                # Source attribution — append below the answer
                sources = item["sources"]
                if sources:
                    src_lines = "\n".join(
                        f"  • [cyan]{s['file']}[/cyan] ({s['score']})"
                        for s in sources
                    )
                    accumulated += f"\n\n[dim]Sources:[/dim]\n{src_lines}"

            elif isinstance(item, dict) and "error" in item:
                # Replace bubble with error styling
                bubble.remove_class("bubble-assistant")
                bubble.add_class("bubble-error")
                accumulated = f"[bold red]Error:[/bold red] {item['error']}"

        # Final render flush
        bubble.update(accumulated)
        await history.scroll_end(animate=False)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    RagChatApp().run()
```

- [ ] **Step 4: Run the smoke tests**

```bash
.venv/bin/python -m pytest tests/test_chat_app.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Run the app manually to verify the layout**

```bash
.venv/bin/python chat_app.py
```

Expected: terminal splits into a header, an empty scroll area, and an input box at the bottom. Press `q` to exit. (The API doesn't need to be running for this visual check.)

- [ ] **Step 6: Commit**

```bash
git add chat_app.py tests/test_chat_app.py
git commit -m "feat: add Textual chat app skeleton with layout and CSS"
```

---

## Task 6: Wire Streaming End-to-End

This task does not require new tests (the streaming logic is already tested in `test_rag_client.py`). It is verified by running the full stack manually.

**Files:**
- No changes needed — `chat_app.py` already has the full streaming implementation from Task 5.

- [ ] **Step 1: Start the RAG API server**

In one terminal:
```bash
cd /home/aharivel/Documents/rag-pdf
.venv/bin/python rag_api.py
```

Expected: `INFO: Application startup complete.` on port 8000.

- [ ] **Step 2: Start the chat app**

In another terminal:
```bash
cd /home/aharivel/Documents/rag-pdf
.venv/bin/python chat_app.py
```

- [ ] **Step 3: Test a first question**

Type: `What is DPDK?` and press Enter.

Expected: a "You" bubble appears, then an "Assistant" bubble fills in token-by-token, ending with a "Sources:" section citing PDF filenames.

- [ ] **Step 4: Test multi-turn (the key feature)**

Type a follow-up question that requires context from the first answer, e.g.: `How does it compare to the kernel network stack?`

Expected: the assistant's answer references the prior exchange (because `CondensePlusContextChatEngine` condenses the history into the retrieval query).

- [ ] **Step 5: Test `/clear`**

Type `/clear`.

Expected: the chat history clears visually. Then ask the follow-up question again — the answer should NOT reference the previous exchange (history was reset).

- [ ] **Step 6: Test error handling**

Stop the API server (`Ctrl+C` in its terminal), then ask a question in the chat app.

Expected: an error-styled bubble appears: `Error: Cannot reach RAG API — is rag_api.py running?`

- [ ] **Step 7: Commit (no code changes, but record the verification)**

```bash
git commit --allow-empty -m "chore: verified streaming end-to-end, multi-turn, /clear, error handling"
```

---

## Task 7: Add `make chat` and `make test` to Makefile

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add the two targets to the Makefile**

Open `Makefile` and make the following changes:

**Add to the `.PHONY` line** (at the top):
```makefile
.PHONY: help setup check index update-index start query clean mcp-config chat test
```

**Add to the `help` echo block** (under `Running`):
```makefile
	@echo "    make chat           Start the Textual RAG chat TUI"
	@echo "    make test           Run all unit tests"
```

**Add the two new targets** (after the `query` target):
```makefile
# ── Chat TUI ──────────────────────────────────────────────────────────────────
chat:
	@echo "Starting PDF RAG Chat TUI..."
	@echo "Tip: make sure 'make start' is running in another terminal.\n"
	$(PYTHON) chat_app.py

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v
```

- [ ] **Step 2: Verify `make test` runs all tests**

```bash
make test
```

Expected: all tests in `tests/` pass, output shows `test_rag_api_helpers.py`, `test_rag_client.py`, and `test_chat_app.py`.

- [ ] **Step 3: Verify `make chat` starts the app**

```bash
make chat
```

Expected: prints the tip message, then launches the Textual app. Press `q` to exit.

- [ ] **Step 4: Commit**

```bash
git add Makefile
git commit -m "chore: add make chat and make test targets"
```

---

## Done

At this point the feature is complete. Here is what you built and what each piece taught you:

| What you built | What it taught |
|---------------|----------------|
| `_build_chat_history()` in `rag_api.py` | LlamaIndex `ChatMessage` / `MessageRole` types |
| `POST /v1/chat/completions/stream` | FastAPI `StreamingResponse`, SSE wire format, sync generator in threadpool |
| `_parse_sse_line()` in `rag_client.py` | SSE parsing, JSON decoding, sentinel patterns |
| `RagClient.send_message()` | `httpx` async streaming, `aiter_lines()`, error propagation without exceptions |
| `RagChatApp.compose()` + CSS | Textual widget tree, CSS layout, `Header`, `VerticalScroll`, `Input` |
| `@work _stream()` | Textual async tasks, direct widget mutation from task, update batching at 20fps |
| `scroll_end()` | Awaiting coroutines on a widget, auto-scroll behaviour |
| `/clear` command | Widget removal (`remove_children()`), client state reset |
