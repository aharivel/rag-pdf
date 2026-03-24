import os
from pathlib import Path
from dotenv import load_dotenv

# Always load .env relative to this file, regardless of working directory
_HERE = Path(__file__).parent
load_dotenv(_HERE / ".env")

# Separate URLs so LLM and embeddings can run on different machines
OLLAMA_LLM_URL   = os.getenv("OLLAMA_LLM_URL",   "http://192.168.1.XXX:11434")
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL",  "http://localhost:11434")
LLM_MODEL        = os.getenv("LLM_MODEL",   "qwen3:8b")
EMBED_MODEL       = os.getenv("EMBED_MODEL", "nomic-embed-text-v2-moe")
CHROMA_PATH     = os.getenv("CHROMA_PATH", str(_HERE / "chroma_db"))
CHROMA_COLLECTION = "pdf_knowledge"
TOP_K           = int(os.getenv("TOP_K", "5"))
OLLAMA_NUM_CTX  = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

# Per-model context window overrides — JSON dict, e.g.:
# OLLAMA_NUM_CTX_MAP={"hf.co/unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF:Q4_K_M": 1024}
import json as _json
_raw = os.getenv("OLLAMA_NUM_CTX_MAP", "{}")
try:
    OLLAMA_NUM_CTX_MAP: dict[str, int] = _json.loads(_raw)
except _json.JSONDecodeError:
    OLLAMA_NUM_CTX_MAP = {}


def get_num_ctx(model: str) -> int:
    """Return the context window for a specific model, falling back to OLLAMA_NUM_CTX."""
    return OLLAMA_NUM_CTX_MAP.get(model, OLLAMA_NUM_CTX)
CHUNK_SIZE       = 512
CHUNK_OVERLAP    = 100
EMBED_BATCH_SIZE = 10

# Base PDF directory
PDF_BASE = os.getenv("PDF_BASE", "/home/youruser/Documents/pdf")

# Folders to index — comment out any you don't want
PDF_FOLDERS = [
    f"{PDF_BASE}/AI",
    f"{PDF_BASE}/go",
    f"{PDF_BASE}/devops",
    f"{PDF_BASE}/linux",
    # f"{PDF_BASE}/virtualization",
    # f"{PDF_BASE}/networking",
    # f"{PDF_BASE}/hardware",
    # f"{PDF_BASE}/power-performance",
    f"{PDF_BASE}/cheatsheets",
    f"{PDF_BASE}/red-hat",
    # f"{PDF_BASE}/misc",
]
