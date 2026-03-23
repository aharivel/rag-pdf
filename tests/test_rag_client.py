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

    class FakeStream:
        async def __aenter__(self):
            return mock_response
        async def __aexit__(self, *a):
            pass

    def fake_stream(*args, **kwargs):
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
