"""
CLI tool to query your PDF knowledge base.

Usage:
  python query_rag.py "How does DPDK handle packet processing?"
  python query_rag.py "Explain KVM memory management"
"""
import sys
import chromadb

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

import config

console = Console()


def load_index():
    Settings.embed_model = OllamaEmbedding(
        model_name=config.EMBED_MODEL,
        base_url=config.OLLAMA_EMBED_URL,
    )
    Settings.llm = Ollama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_LLM_URL,
        request_timeout=120.0,
    )

    chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    chroma_collection = chroma_client.get_collection(config.CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )


def query(question: str):
    console.print(Panel.fit(f"[bold blue]{question}[/bold blue]", title="Question"))

    index = load_index()
    query_engine = index.as_query_engine(
        similarity_top_k=config.TOP_K,
        streaming=False,
    )

    console.print("\n[dim]Retrieving relevant chunks and querying LLM...[/dim]\n")
    response = query_engine.query(question)

    console.print(Rule("[bold green]Answer[/bold green]"))
    console.print(Markdown(str(response)))

    console.print(Rule("[bold yellow]Sources[/bold yellow]"))
    seen = set()
    for node in response.source_nodes:
        file_name = node.metadata.get("file_name", "unknown")
        score = node.score or 0
        if file_name not in seen:
            seen.add(file_name)
            console.print(f"  • [cyan]{file_name}[/cyan]  (score: {score:.3f})")
    console.print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[red]Usage: python query_rag.py \"your question here\"[/red]")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    query(question)
