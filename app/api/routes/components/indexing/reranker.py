"""NVIDIA NIM reranker — reorders retrieved documents by relevance to a query.

Uses the langchain_nvidia_ai_endpoints ``NVIDIARerank`` class, which calls
the NVIDIA-hosted reranking NIM at ``https://integrate.api.nvidia.com/v1``.

The default model is ``nvidia/nv-rerankqa-mistral-4b-v3`` — a cross-encoder
based on Mistral-7B (first 16 layers) fine-tuned for QA relevance scoring.
"""

from __future__ import annotations

import asyncio
import logging

from .config import get_settings

logger = logging.getLogger(__name__)


def make_reranker():
    """Lazily instantiate ``NVIDIARerank`` (import-safe without creds)."""
    from langchain_nvidia_ai_endpoints import NVIDIARerank

    s = get_settings()
    return NVIDIARerank(
        model=s.nvidia_rerank_model,
        api_key=s.nvidia_api_key_str,
        top_n=s.rerank_top_n,
    )


async def rerank_documents(
    query: str,
    documents: list,
    reranker=None,
) -> list:
    """
    Rerank LangChain ``Document`` objects by relevance to ``query``.

    Returns documents reordered by relevance (most relevant first), truncated
    to ``rerank_top_n``.  Each document's ``metadata`` gets a ``relevance_score``.

    The ``compress_documents`` call is synchronous — offloaded to a thread.
    """
    ranker = reranker or make_reranker()
    if not documents:
        return []

    reranked = await asyncio.to_thread(
        ranker.compress_documents,
        documents=documents,
        query=query,
    )

    logger.info(
        "rerank_documents: %d input → %d output (top_n=%d)",
        len(documents), len(reranked), get_settings().rerank_top_n,
    )
    return reranked
