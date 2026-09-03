"""
indexing — Modular chunking + embedding + reranking + retrieval pipeline.

Public API:
    # Indexing (ingest-time)
    from app.api.routes.components.indexing import index_chunks_to_azure
    result = await index_chunks_to_azure(chunks, notebook_id="...", dry_run=False)

    # Retrieval + grounded answer (query-time)
    from app.api.routes.components.indexing import retrieve_for_notebook, generate_grounded_answer
    results = await retrieve_for_notebook(query="...", notebook_id="...")
    answer = await generate_grounded_answer(query="...", results=results)
"""

from .pipeline import index_chunks_to_azure, get_vectorstore
from .config import get_settings, deterministic_id
from .image_embeddings import OpenRouterImageEmbedder
from .reranker import make_reranker, rerank_documents
from .retrieval import (
    retrieve,
    retrieve_for_notebook,
    retrieve_for_source,
    RetrievalResult,
)
from .answerer import generate_grounded_answer

__all__ = [
    # Indexing
    "index_chunks_to_azure",
    "get_vectorstore",
    # Config
    "get_settings",
    "deterministic_id",
    # Image embeddings
    "OpenRouterImageEmbedder",
    # Reranker
    "make_reranker",
    "rerank_documents",
    # Retrieval
    "retrieve",
    "retrieve_for_notebook",
    "retrieve_for_source",
    "RetrievalResult",
    # Answer generation
    "generate_grounded_answer",
]
