"""Async HTTP client for the PDF RAG streaming API.

Owns conversation history and SSE stream parsing.
Does not import anything from textual.
"""
from __future__ import annotations

import datetime
import json
import os
from collections.abc import AsyncGenerator
from pathlib import Path

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
        self.session_log: list[dict] = []
        self.started_at = datetime.datetime.now()
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(None, connect=5.0),
        )

    def clear_history(self) -> None:
        """Reset conversation history (used by /clear command)."""
        self.conversation_history.clear()

    def record_exchange(
        self,
        question: str,
        answer: str,
        sources: list[dict],
        stats: dict,
    ) -> None:
        """Append a completed exchange to the session log."""
        self.session_log.append({
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "question": question,
            "answer": answer,
            "sources": sources,
            "stats": stats,
        })

    def save_session(self, chats_dir: str = "chats") -> str:
        """Write the session log to a JSON file. Returns the file path."""
        path = Path(chats_dir)
        path.mkdir(exist_ok=True)
        model_slug = (
            self.session_log[0]["stats"]["model"].replace(":", "-")
            if self.session_log else "unknown"
        )
        ts = self.started_at.strftime("%Y-%m-%d_%H-%M-%S")
        filepath = path / f"{ts}_{model_slug}.json"
        data = {
            "model": model_slug,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "exchanges": self.session_log,
        }
        filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        return str(filepath)

    async def send_message(self, text: str) -> AsyncGenerator[str | dict, None]:
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
                    self.conversation_history.pop()  # roll back before yield
                    yield {"error": f"LLM unavailable (HTTP {response.status_code})"}
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
                    elif "stats" in parsed:
                        yield parsed  # {"stats": {...}}
                    elif "error" in parsed:
                        yield parsed  # {"error": "..."}
                        break

        except httpx.ConnectError:
            self.conversation_history.pop()  # roll back before yield
            yield {"error": "Cannot reach RAG API — is rag_api.py running?"}
            return
        except httpx.ConnectTimeout:
            self.conversation_history.pop()  # roll back before yield
            yield {"error": "Connection timed out — is rag_api.py running?"}
            return
        except httpx.RemoteProtocolError:
            if not accumulated:
                self.conversation_history.pop()  # no partial reply to preserve
            yield {"error": "Stream interrupted — partial response shown"}

        if accumulated:
            self.conversation_history.append(
                {"role": "assistant", "content": accumulated}
            )
