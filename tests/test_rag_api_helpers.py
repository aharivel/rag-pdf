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
