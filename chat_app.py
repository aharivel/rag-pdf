"""Textual TUI chat app for the PDF RAG pipeline.

Usage:
    python chat_app.py
    make chat

Architecture:
    This file is pure UI. All network and RAG logic lives in rag_client.py.
    Data flows: Input → @work async task → rag_client.send_message() → Static.update()
"""
from __future__ import annotations

import re
import shutil
import subprocess
import time
from functools import partial

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import Hit, Hits, Provider
from textual.containers import VerticalScroll
from textual.widgets import Header, Input, Markdown, Static

from rag_client import RagClient


# ── Snip helpers ──────────────────────────────────────────────────────────────

_SNIP_RE = re.compile(
    r'<snip\s+category="([^"]+)"\s+headline="([^"]+)"(?:\s+lang="([^"]*)")?>'
    r'(.*?)</snip>',
    re.DOTALL,
)
_CODE_BLOCK_RE = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)


def _extract_snips(text: str) -> tuple[str, list[dict]]:
    """Strip <snip> tags from text. Returns (clean_text, list_of_snip_dicts)."""
    snips = []
    for m in _SNIP_RE.finditer(text):
        snips.append({
            "category": m.group(1),
            "headline": m.group(2),
            "lang": m.group(3) or "",
            "code": m.group(4).strip(),
        })
    return _SNIP_RE.sub("", text).strip(), snips


def _extract_first_code_block(text: str) -> tuple[str, str]:
    """Return (lang, code) of the first fenced code block, or ('', '') if none found."""
    m = _CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1), m.group(2).strip()
    return "", ""


def _call_snip(category: str, headline: str, code: str, lang: str) -> bool:
    """Call `snip add`. Returns True on success, False if snip not found or failed."""
    snip_bin = shutil.which("snip")
    if not snip_bin:
        return False
    cmd = [snip_bin, "add", category, headline, "--snippet", code]
    if lang:
        cmd += ["--lang", lang]
    return subprocess.run(cmd, capture_output=True).returncode == 0


# ── Command palette — model switcher ──────────────────────────────────────────

class ModelProvider(Provider):
    """Lists available Ollama models in the command palette (Ctrl+P)."""

    async def startup(self) -> None:
        self._models = await self.app._client.list_models()

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for name in self._models:
            score = matcher.match(name)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(name),
                    partial(self.app._select_model, name),
                    text=name,
                    help="Switch to this model",
                )


# ── CSS ───────────────────────────────────────────────────────────────────────

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

.bubble-stats {
    color: $text-muted;
    margin: 0 0 1 2;
    padding: 0;
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
    COMMANDS = App.COMMANDS | {ModelProvider}

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("ctrl+p", "command_palette", "Models", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._client = RagClient()

    async def _select_model(self, model: str) -> None:
        """Called by ModelProvider when the user picks a model."""
        self._client.set_model(model)
        self.sub_title = model
        self.notify(f"Switched to {model}", timeout=4)

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="history")
        yield Input(placeholder="Ask your PDF library… (/clear  /save  /mode  /quit)")

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
        if text == "/save":
            await self._do_save()
            return
        if text == "/quit":
            await self._do_quit()
            return
        if text.startswith("/mode"):
            await self._do_mode(text)
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

    _MODES = ["condense_plus_context", "context", "simple"]

    async def _do_mode(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        if len(parts) == 2 and parts[1] in self._MODES:
            mode = parts[1]
        else:
            # cycle to next mode
            current = self._client.chat_mode
            idx = self._MODES.index(current) if current in self._MODES else 0
            mode = self._MODES[(idx + 1) % len(self._MODES)]
        self._client.set_mode(mode)
        self.notify(f"Chat mode → {mode}", timeout=5)

    async def _do_clear(self) -> None:
        self._client.clear_history()
        history = self.query_one("#history", VerticalScroll)
        await history.remove_children()

    async def _do_save(self) -> None:
        if not self._client.session_log:
            self.notify("Nothing to save yet.", severity="warning")
            return
        path = self._client.save_session()
        self.notify(f"Saved → {path}", timeout=6)

    async def _do_quit(self) -> None:
        if self._client.session_log:
            self._client.save_session()
        self.exit()

    @work
    async def _stream(self, text: str, bubble: Markdown) -> None:
        history = self.query_one("#history", VerticalScroll)
        accumulated = ""   # display text (tokens + formatted sources)
        answer_text = ""   # clean LLM answer only (saved to log)
        sources_raw: list[dict] = []
        last_render = time.monotonic()
        RENDER_INTERVAL = 0.05  # 50ms → ~20 fps
        stats: dict | None = None

        async for item in self._client.send_message(text):
            if isinstance(item, str):
                accumulated += item
                answer_text += item
                now = time.monotonic()
                if now - last_render >= RENDER_INTERVAL:
                    bubble.update(accumulated)
                    history.scroll_end(animate=False)
                    last_render = now

            elif isinstance(item, dict) and "sources" in item:
                sources_raw = item["sources"]
                if sources_raw:
                    src_lines = "\n".join(
                        f"- `{s['file']}` ({s['score']})"
                        for s in sources_raw
                    )
                    accumulated += f"\n\n---\n**Sources:**\n{src_lines}"

            elif isinstance(item, dict) and "stats" in item:
                stats = item["stats"]

            elif isinstance(item, dict) and "error" in item:
                bubble.remove_class("bubble-assistant")
                bubble.add_class("bubble-error")
                accumulated = f"**Error:** {item['error']}"

        bubble.update(accumulated)

        if stats:
            info = (
                f"{stats['model']}  ·  "
                f"{stats['tokens']} tokens  ·  "
                f"{stats['tokens_per_sec']} tok/s  ·  "
                f"{stats['eval_s']}s gen  ·  "
                f"{stats['total_s']}s total"
            )
            await history.mount(Static(info, classes="bubble-stats"))
            self._client.record_exchange(text, answer_text, sources_raw, stats)

        history.scroll_end(animate=False)


if __name__ == "__main__":
    RagChatApp().run()
