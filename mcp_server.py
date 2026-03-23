"""
MCP server exposing your PDF library as a tool for Claude Code.

Claude Code spawns this as a subprocess (stdio transport).
It embeds questions via Ollama, searches ChromaDB, and returns
relevant chunks — Claude then synthesizes the answer.

Configure in ~/.claude/settings.json:
{
  "mcpServers": {
    "pdf-rag": {
      "command": "/home/youruser/Documents/rag-pdf/.venv/bin/python",
      "args": ["/home/youruser/Documents/rag-pdf/mcp_server.py"]
    }
  }
}
"""
import asyncio
import chromadb
import mcp.types as types

from mcp.server import Server
from mcp.server.stdio import stdio_server
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

import config

server = Server("pdf-rag")

# Retriever loaded once and reused across calls
_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        Settings.embed_model = OllamaEmbedding(
            model_name=config.EMBED_MODEL,
            base_url=config.OLLAMA_EMBED_URL,
            embed_batch_size=config.EMBED_BATCH_SIZE,
        )
        # LLM is set but won't be called — we use retriever only
        Settings.llm = Ollama(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_LLM_URL,
        )

        chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        chroma_collection = chroma_client.get_collection(config.CHROMA_COLLECTION)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
        _retriever = index.as_retriever(similarity_top_k=config.TOP_K)

    return _retriever


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="query_pdf_library",
            description=(
                "Search the user's personal PDF library using semantic similarity. "
                "The library contains technical books and documentation on topics including "
                "Go programming, AI/LLMs, DevOps, Linux, networking, virtualization, and more. "
                "Returns the most relevant text chunks with their source filenames. "
                "Use this when answering questions that could be informed by the user's books."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question or topic to search for in the PDF library",
                    }
                },
                "required": ["question"],
            },
        ),
        types.Tool(
            name="list_indexed_folders",
            description="List which PDF folders are currently indexed in the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

    if name == "list_indexed_folders":
        folders = "\n".join(f"  • {f}" for f in config.PDF_FOLDERS)
        total = chromadb.PersistentClient(path=config.CHROMA_PATH) \
            .get_collection(config.CHROMA_COLLECTION).count()
        return [types.TextContent(
            type="text",
            text=f"Indexed folders:\n{folders}\n\nTotal chunks in database: {total}",
        )]

    if name == "query_pdf_library":
        question = arguments.get("question", "")
        if not question:
            return [types.TextContent(type="text", text="No question provided.")]

        try:
            retriever = get_retriever()
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error loading index: {e}\nMake sure you have run 'make index' first.",
            )]

        nodes = retriever.retrieve(question)

        if not nodes:
            return [types.TextContent(
                type="text",
                text="No relevant content found in the PDF library for this question.",
            )]

        # Format chunks for Claude to synthesize
        parts = [f"Found {len(nodes)} relevant chunks from your PDF library:\n"]
        for i, node in enumerate(nodes, 1):
            source = node.metadata.get("file_name", "unknown")
            score = round(node.score or 0, 3)
            text = node.text.strip()
            parts.append(f"--- Chunk {i} | Source: {source} | Relevance: {score} ---\n{text}\n")

        return [types.TextContent(type="text", text="\n".join(parts))]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
