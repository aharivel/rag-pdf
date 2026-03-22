# PDF RAG Setup Guide

## Architecture

```
[Linux Laptop] ←─ 2.5Gbit ─→ [Windows PC]
  • PDFs                         • Ollama (GPU)
  • ChromaDB                     • llama3.1:8b
  • LlamaIndex                   • nomic-embed-text
  • FastAPI (RAG API)
  • Open WebUI
```

---

## Step 1 — Windows PC: Install & configure Ollama

1. Download and install Ollama from https://ollama.com
2. Allow Ollama to listen on the network (required for laptop access):
   - Open **System Properties → Environment Variables**
   - Add a new **System variable**: `OLLAMA_HOST` = `0.0.0.0`
   - Restart Ollama (or reboot)
3. Open Windows Firewall → Allow inbound on **port 11434**
4. Pull the required models (open PowerShell or CMD):
   ```
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```
5. Find your PC's local IP (run in PowerShell):
   ```
   ipconfig
   ```
   Look for the IPv4 address on your LAN adapter (e.g. `192.168.1.42`)

---

## Step 2 — Linux Laptop: Configure the project

```bash
cd /home/youruser/Documents/rag-pdf

# Copy the example env file
cp .env.example .env

# Edit .env and set your Windows PC IP
nano .env   # change OLLAMA_BASE_URL=http://192.168.1.XXX:11434

# Create a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Step 3 — Verify connection to Windows PC

```bash
source .venv/bin/activate
python check_connection.py
```

Expected output:
```
  [OK]  Ollama is reachable
  [OK]  llama3.1:8b found
  [OK]  nomic-embed-text found
  All good! You can now run index_pdfs.py
```

---

## Step 4 — Index your PDFs

This embeds all your PDFs into ChromaDB. It will take a while (~1-3 min per
100 pages depending on network speed). Only needs to run once.

```bash
python index_pdfs.py
```

To add new PDFs later without re-indexing everything:
```bash
python index_pdfs.py --update
```

---

## Step 5 — Test with CLI query tool

```bash
python query_rag.py "How does DPDK handle packet processing?"
python query_rag.py "What is QLoRA and how does it reduce VRAM usage?"
python query_rag.py "Explain goroutines and channels in Go"
python query_rag.py "How does KVM handle memory virtualization?"
```

Each answer shows the sources (which PDF it came from).

---

## Step 6 — Start the RAG API server

```bash
python rag_api.py
```

The API runs at `http://localhost:8000`. Keep this terminal open.

---

## Step 7 — Connect Open WebUI

1. Open Open WebUI in your browser
2. Go to **Settings → Connections → OpenAI API**
3. Set:
   - **URL**: `http://localhost:8000/v1`
   - **API Key**: `dummy` (any non-empty string)
4. Save → you should see **pdf-rag** appear in the model selector
5. Select **pdf-rag** and start chatting with your PDF library!

---

## Project structure

```
rag-pdf/
├── .env                  # Your config (not committed)
├── .env.example          # Template
├── config.py             # Loads settings from .env
├── requirements.txt      # Python dependencies
├── check_connection.py   # Verify Ollama on Windows PC is reachable
├── index_pdfs.py         # Index PDFs → ChromaDB
├── query_rag.py          # CLI query tool (for testing)
├── rag_api.py            # FastAPI server (connects to Open WebUI)
└── chroma_db/            # Vector database (created after indexing)
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot reach Ollama` | Check OLLAMA_HOST=0.0.0.0 on Windows, check firewall port 11434 |
| `Collection not found` | Run `index_pdfs.py` first |
| Slow responses | Normal — LLM inference on RTX 3060 takes 5-15s per query |
| Open WebUI can't see pdf-rag model | Make sure `rag_api.py` is running and URL is `http://localhost:8000/v1` |
| Bad answers | Try increasing TOP_K in .env (e.g. TOP_K=8), or re-index with smaller chunks |
