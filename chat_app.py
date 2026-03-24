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
from textual.containers import VerticalScroll
from textual.widgets import Header, Input, Markdown, Static

from rag_client import RagClient


CSS = """
Screen {
    background: $surface;
}

#history {
    height: 1fr;
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


class RagChatApp(App):
    """PDF RAG Chat — a Textual TUI for your local knowledge base."""

    TITLE = "PDF RAG Chat"
    CSS = CSS

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._client = RagClient()

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="history")
        yield Input(placeholder="Ask your PDF library… (/clear to reset, /quit to exit)")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.clear()

        if not text:
            return

        if text == "/clear":
            await self._do_clear()
            return
        if text == "/quit":
            self.exit()
            return

        await self._submit_query(text)

    async def _submit_query(self, text: str) -> None:
        history = self.query_one("#history", VerticalScroll)

        await history.mount(Static(f"[bold]You[/bold]", classes="bubble-label"))
        await history.mount(Static(text, classes="bubble-user"))
        await history.mount(Static("[bold]Assistant[/bold]", classes="bubble-label"))
        assistant_bubble = Markdown("", classes="bubble-assistant")
        await history.mount(assistant_bubble)

        history.scroll_end(animate=False)

        self._stream(text, assistant_bubble)

    async def _do_clear(self) -> None:
        self._client.clear_history()
        history = self.query_one("#history", VerticalScroll)
        await history.remove_children()

    @work
    async def _stream(self, text: str, bubble: Markdown) -> None:
        history = self.query_one("#history", VerticalScroll)
        accumulated = ""
        last_render = time.monotonic()
        RENDER_INTERVAL = 0.05  # 50ms → ~20 fps

        async for item in self._client.send_message(text):
            if isinstance(item, str):
                accumulated += item
                now = time.monotonic()
                if now - last_render >= RENDER_INTERVAL:
                    bubble.update(accumulated)
                    history.scroll_end(animate=False)
                    last_render = now

            elif isinstance(item, dict) and "sources" in item:
                sources = item["sources"]
                if sources:
                    src_lines = "\n".join(
                        f"- `{s['file']}` ({s['score']})"
                        for s in sources
                    )
                    accumulated += f"\n\n---\n**Sources:**\n{src_lines}"

            elif isinstance(item, dict) and "error" in item:
                bubble.remove_class("bubble-assistant")
                bubble.add_class("bubble-error")
                accumulated = f"**Error:** {item['error']}"

        bubble.update(accumulated)
        history.scroll_end(animate=False)


if __name__ == "__main__":
    RagChatApp().run()
