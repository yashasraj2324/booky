"""
indexing — Modular chunking + embedding + Azure AI Search pipeline.

Public API:
    from app.api.routes.components.indexing import index_chunks_to_azure

    result = await index_chunks_to_azure(chunks, dry_run=False)
"""

from .pipeline import index_chunks_to_azure, get_vectorstore
from .config import get_settings, deterministic_id
from .image_embeddings import OpenRouterImageEmbedder

__all__ = [
    "index_chunks_to_azure",
    "get_vectorstore",
    "get_settings",
    "deterministic_id",
    "OpenRouterImageEmbedder",
]
