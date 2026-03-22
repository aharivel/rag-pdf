.PHONY: help setup check index update-index start query clean mcp-config

PYTHON = .venv/bin/python
PIP    = .venv/bin/pip

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  PDF RAG — available commands"
	@echo ""
	@echo "  Setup"
	@echo "    make setup          Create venv and install dependencies"
	@echo "    make check          Verify Ollama on Windows PC is reachable"
	@echo ""
	@echo "  Indexing"
	@echo "    make index          Full re-index (clears DB and reindexes all folders)"
	@echo "    make update-index   Only index PDFs not yet in the database"
	@echo ""
	@echo "  Running"
	@echo "    make start          Start the RAG API server (Open WebUI endpoint)"
	@echo "    make query Q=\"...\"  Run a one-off query from the terminal"
	@echo ""
	@echo "  MCP (Claude Code integration)"
	@echo "    make mcp-config     Print config snippet for ~/.claude/settings.json"
	@echo ""
	@echo "  Maintenance"
	@echo "    make clean          Delete the ChromaDB vector database"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	@echo "Creating virtual environment..."
	python3 -m venv .venv
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip -q
	$(PIP) install -r requirements.txt
	@[ -f .env ] || (cp .env.example .env && echo "\n  ⚠  .env created from template — edit it with your Windows PC IP before continuing.\n")
	@echo "\nDone. Run 'make check' next."

# ── Preflight check ───────────────────────────────────────────────────────────
check:
	$(PYTHON) check_connection.py

# ── Indexing ──────────────────────────────────────────────────────────────────
index:
	@echo "Running full re-index (this will clear the existing database)..."
	$(PYTHON) index_pdfs.py

update-index:
	@echo "Indexing new PDFs only..."
	$(PYTHON) index_pdfs.py --update

# ── API server ────────────────────────────────────────────────────────────────
start:
	@echo "Starting RAG API on http://0.0.0.0:8000 ..."
	@echo "Add http://<this-laptop-ip>:8000/v1 to Open WebUI as an OpenAI connection."
	@echo "Press Ctrl+C to stop.\n"
	$(PYTHON) rag_api.py

# ── One-off query ─────────────────────────────────────────────────────────────
query:
ifndef Q
	@echo "Usage: make query Q=\"your question here\""
else
	$(PYTHON) query_rag.py "$(Q)"
endif

# ── MCP config ────────────────────────────────────────────────────────────────
mcp-config:
	@echo ""
	@echo "Add this to ~/.claude/settings.json under \"mcpServers\":"
	@echo ""
	@echo "  \"pdf-rag\": {"
	@echo "    \"command\": \"$(shell pwd)/.venv/bin/python\","
	@echo "    \"args\": [\"$(shell pwd)/mcp_server.py\"]"
	@echo "  }"
	@echo ""

# ── Maintenance ───────────────────────────────────────────────────────────────
clean:
	@echo "Deleting ChromaDB database..."
	rm -rf chroma_db/
	@echo "Done. Run 'make index' to rebuild."
