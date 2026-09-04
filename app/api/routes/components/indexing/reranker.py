"""NVIDIA NIM reranker — routed through the Pydantic gateway.

Calls POST /v1/ranking on the gateway, which routes to
nvidia/nv-rerankqa-mistral-4b-v3 based on the ``model`` field.
"""

from __future__ import annotations

import asyncio
import logging

from .config import get_settings
from .gateway import get_gateway

logger = logging.getLogger(__name__)


async def rerank_documents(
    query: str,
    documents: list,
    gateway=None,
) -> list:
    """
    Rerank LangChain ``Document`` objects by relevance to ``query``.

    Extracts passage text from each Document, sends to the gateway's
    /v1/ranking endpoint, then reorders the Documents by relevance score.

    Returns documents reordered (most relevant first), truncated to
    ``rerank_top_n``.  Each document's ``metadata`` gets a
    ``relevance_score`` key.
    """
    if not documents:
        return []

    gw = gateway or get_gateway()
    settings = get_settings()

    # Extract passage texts
    passages = [doc.page_content for doc in documents]

    # Call the gateway's reranking endpoint (synchronous → thread)
    rankings = await asyncio.to_thread(
        gw.rerank, query, passages, settings.rerank_top_n
    )

    # Reorder documents by ranking
    reranked = []
    for r in rankings:
        idx = r["index"]
        if 0 <= idx < len(documents):
            doc = documents[idx]
            doc.metadata["relevance_score"] = r["relevance_score"]
            reranked.append(doc)

    logger.info(
        "rerank_documents: %d input → %d output (top_n=%d)",
        len(documents), len(reranked), settings.rerank_top_n,
    )
    return reranked
