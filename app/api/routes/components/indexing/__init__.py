"""
indexing — Modular pipeline: Pydantic gateway + Cosmos DB + LangGraph retrieval.

Public API:
    # Indexing (ingest-time)
    from app.api.routes.components.indexing import index_chunks_to_cosmos
    result = await index_chunks_to_cosmos(chunks, notebook_id="...", dry_run=False)

    # Retrieval + grounded answer (query-time)
    from app.api.routes.components.indexing import retrieve_for_notebook, generate_grounded_answer
    results = await retrieve_for_notebook(query="...", notebook_id="...")
    answer = await generate_grounded_answer(query="...", results=results)
"""

from .pipeline import index_chunks_to_cosmos
from .config import get_settings, deterministic_id, GatewayClient, get_gateway
from .reranker import rerank_documents
from .retrieval import (
    retrieve,
    retrieve_for_notebook,
    RetrievalResult,
)
from .answerer import generate_grounded_answer

__all__ = [
    # Indexing
    "index_chunks_to_cosmos",
    # Config
    "get_settings",
    "deterministic_id",
    # Gateway
    "GatewayClient",
    "get_gateway",
    # Reranker
    "rerank_documents",
    # Retrieval
    "retrieve",
    "retrieve_for_notebook",
    "RetrievalResult",
    # Answer generation
    "generate_grounded_answer",
]
