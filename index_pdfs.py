"""
Index PDFs from selected folders (configured in config.py) into ChromaDB.

Usage:
  python index_pdfs.py           # full re-index (clears existing data)
  python index_pdfs.py --update  # only index PDFs not yet in the database

To change which folders are indexed, edit PDF_FOLDERS in config.py.
"""
import sys
import argparse
import chromadb
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from rich.console import Console
from rich.panel import Panel

import config

console = Console()


def setup_settings():
    Settings.embed_model = OllamaEmbedding(
        model_name=config.EMBED_MODEL,
        base_url=config.OLLAMA_EMBED_URL,
        embed_batch_size=config.EMBED_BATCH_SIZE,
    )
    Settings.llm = Ollama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_LLM_URL,
        request_timeout=120.0,
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )


def get_indexed_files(collection) -> set:
    """Return set of file paths already in the collection."""
    try:
        results = collection.get(include=["metadatas"])
        return {m.get("file_path", "") for m in results["metadatas"] if m}
    except Exception:
        return set()


def index_pdfs(update_mode: bool = False):
    folders_display = "\n".join(f"  • {f}" for f in config.PDF_FOLDERS)
    console.print(Panel.fit(
        f"[bold blue]PDF RAG Indexer[/bold blue]\n"
        f"Folders    :\n{folders_display}\n"
        f"ChromaDB   : {config.CHROMA_PATH}\n"
        f"Ollama     : {config.OLLAMA_BASE_URL}\n"
        f"LLM model  : {config.LLM_MODEL}\n"
        f"Embed model: {config.EMBED_MODEL}\n"
        f"Mode       : {'update (new files only)' if update_mode else 'full re-index'}",
    ))

    setup_settings()

    chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)

    if not update_mode:
        # Wipe and recreate the collection
        try:
            chroma_client.delete_collection(config.CHROMA_COLLECTION)
            console.print("[yellow]Cleared existing collection.[/yellow]")
        except Exception:
            pass

    chroma_collection = chroma_client.get_or_create_collection(config.CHROMA_COLLECTION)

    # Find all PDFs across selected folders
    all_pdfs = []
    for folder in config.PDF_FOLDERS:
        folder_path = Path(folder)
        if not folder_path.exists():
            console.print(f"[yellow]Warning: folder not found, skipping: {folder}[/yellow]")
            continue
        found = list(folder_path.rglob("*.pdf"))
        # Skip files in old/ subfolders
        found = [p for p in found if "old" not in p.parts]
        console.print(f"  [dim]{folder}[/dim] → {len(found)} PDFs")
        all_pdfs.extend(found)
    console.print(f"\nTotal: [bold]{len(all_pdfs)}[/bold] PDFs to process.")

    if update_mode:
        indexed = get_indexed_files(chroma_collection)
        all_pdfs = [p for p in all_pdfs if str(p) not in indexed]
        console.print(f"[bold]{len(all_pdfs)}[/bold] new PDFs to index.")

    if not all_pdfs:
        console.print("[green]Nothing to index. Database is up to date.[/green]")
        return

    # Load documents
    console.print("\n[bold blue]Loading PDFs...[/bold blue]")
    reader = SimpleDirectoryReader(
        input_files=[str(p) for p in all_pdfs],
        filename_as_id=True,
    )
    documents = reader.load_data(show_progress=True)
    console.print(f"Loaded [bold]{len(documents)}[/bold] document pages.")

    # Index into ChromaDB
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    console.print("\n[bold blue]Embedding and indexing (this will take a while)...[/bold blue]")
    console.print("[dim]Tip: embeddings are computed on your Windows PC via Ollama.[/dim]\n")

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    total = chroma_collection.count()
    console.print(f"\n[bold green]✓ Done! {total} chunks stored in ChromaDB.[/bold green]")
    console.print(f"[dim]Run python query_rag.py \"your question\" to test.[/dim]\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="Only index new PDFs")
    args = parser.parse_args()
    index_pdfs(update_mode=args.update)
