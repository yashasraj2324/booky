"""Text embeddings via Azure OpenAI (text-embedding-3-large, 3072 dims)."""

from __future__ import annotations

import asyncio
import logging

from .config import get_settings

logger = logging.getLogger(__name__)


def make_text_embedder():
    """Lazily instantiate AzureOpenAIEmbeddings (import-safe without creds)."""
    from langchain_openai import AzureOpenAIEmbeddings

    s = get_settings()
    return AzureOpenAIEmbeddings(
        azure_endpoint=s.azure_openai_endpoint,
        api_key=s.openai_api_key_str,
        azure_deployment=s.azure_openai_embedding_deployment,
        api_version=s.azure_openai_api_version,
    )


async def embed_texts_batched(
    texts: list[str],
    embedder=None,
) -> list[list[float]]:
    """
    Embed texts in batches.  ``embed_documents`` is synchronous, so each
    batch is offloaded to a thread via ``asyncio.to_thread`` to avoid
    blocking the event loop.
    """
    emb = embedder or make_text_embedder()
    batch_size = get_settings().embed_batch_size
    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batch_vectors = await asyncio.to_thread(emb.embed_documents, batch)
        vectors.extend(batch_vectors)
        logger.debug("embed_texts_batched: %d/%d done", len(vectors), len(texts))

    return vectors
