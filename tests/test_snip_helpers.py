import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch
import pytest

from chat_app import _extract_snips, _extract_first_code_block, _call_snip


# ── _extract_snips ────────────────────────────────────────────────────────────

def test_extract_snips_strips_tag():
    text = (
        'Use this:\n'
        '<snip category="linux" headline="List files" lang="bash">\n'
        'ls -la\n'
        '</snip>\n'
        'Done.'
    )
    clean, snips = _extract_snips(text)
    assert "<snip" not in clean
    assert "ls -la" not in clean
    assert "Done." in clean
    assert len(snips) == 1
    assert snips[0] == {
        "category": "linux", "headline": "List files", "lang": "bash", "code": "ls -la"
    }


def test_extract_snips_no_lang():
    text = '<snip category="go" headline="Hello">fmt.Println("hi")</snip>'
    _, snips = _extract_snips(text)
    assert snips[0]["lang"] == ""


def test_extract_snips_multiple():
    text = (
        '<snip category="linux" headline="A" lang="bash">cmd1</snip>\n'
        'text\n'
        '<snip category="go" headline="B" lang="go">cmd2</snip>'
    )
    _, snips = _extract_snips(text)
    assert len(snips) == 2
    assert snips[0]["code"] == "cmd1"
    assert snips[1]["code"] == "cmd2"


def test_extract_snips_no_tags():
    text = "Just regular text with no tags."
    clean, snips = _extract_snips(text)
    assert clean == text
    assert snips == []


def test_extract_snips_multiline_code():
    text = '<snip category="bash" headline="Multi" lang="bash">\nline1\nline2\n</snip>'
    _, snips = _extract_snips(text)
    assert snips[0]["code"] == "line1\nline2"


# ── _extract_first_code_block ─────────────────────────────────────────────────

def test_extract_first_code_block_with_lang():
    text = "Intro\n```bash\nls -la\n```\nOutro"
    lang, code = _extract_first_code_block(text)
    assert lang == "bash"
    assert code == "ls -la"


def test_extract_first_code_block_no_lang():
    text = "```\nsome code\n```"
    lang, code = _extract_first_code_block(text)
    assert lang == ""
    assert code == "some code"


def test_extract_first_code_block_no_fence_returns_empty():
    # When no fence exists, return ("", "") — do NOT return the text itself.
    # This allows callers to distinguish "no code block" from "empty code block".
    lang, code = _extract_first_code_block("No code block here")
    assert lang == ""
    assert code == ""


def test_extract_first_code_block_returns_first_only():
    text = "```bash\nfirst\n```\n```python\nsecond\n```"
    lang, code = _extract_first_code_block(text)
    assert code == "first"


# ── _call_snip ────────────────────────────────────────────────────────────────

def test_call_snip_returns_false_when_not_found():
    with patch("chat_app.shutil.which", return_value=None):
        assert _call_snip("linux", "headline", "code", "bash") is False


def test_call_snip_returns_true_on_success():
    with patch("chat_app.shutil.which", return_value="/usr/local/bin/snip"), \
         patch("chat_app.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        assert _call_snip("linux", "headline", "ls -la", "bash") is True
        cmd = mock_run.call_args[0][0]
        assert cmd == ["/usr/local/bin/snip", "add", "linux", "headline",
                       "--snippet", "ls -la", "--lang", "bash"]


def test_call_snip_omits_lang_when_empty():
    with patch("chat_app.shutil.which", return_value="/usr/local/bin/snip"), \
         patch("chat_app.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        _call_snip("linux", "headline", "code", "")
        cmd = mock_run.call_args[0][0]
        assert "--lang" not in cmd


def test_call_snip_returns_false_on_nonzero_exit():
    with patch("chat_app.shutil.which", return_value="/usr/local/bin/snip"), \
         patch("chat_app.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        assert _call_snip("linux", "headline", "code", "") is False
