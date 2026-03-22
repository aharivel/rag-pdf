import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.1.XXX:11434")
LLM_MODEL       = os.getenv("LLM_MODEL", "qwen3:8b")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "nomic-embed-text-v2-moe")
CHROMA_PATH     = os.getenv("CHROMA_PATH", "./chroma_db")
CHROMA_COLLECTION = "pdf_knowledge"
TOP_K           = int(os.getenv("TOP_K", "5"))
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
    # f"{PDF_BASE}/linux",
    # f"{PDF_BASE}/virtualization",
    # f"{PDF_BASE}/networking",
    # f"{PDF_BASE}/hardware",
    # f"{PDF_BASE}/power-performance",
    # f"{PDF_BASE}/cheatsheets",
    # f"{PDF_BASE}/red-hat",
    # f"{PDF_BASE}/misc",
]
