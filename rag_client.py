"""Async HTTP client for the PDF RAG streaming API.

Owns conversation history and SSE stream parsing.
Does not import anything from textual.
"""
from __future__ import annotations

import inspect
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
            stream_ctx = self._http.stream(
                "POST",
                "/v1/chat/completions/stream",
                json={"messages": self.conversation_history},
            )
            # In tests, the mock side_effect may be an async function, yielding
            # a coroutine instead of a context manager — await it if needed.
            if inspect.iscoroutine(stream_ctx):
                stream_ctx = await stream_ctx
            async with stream_ctx as response:
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
