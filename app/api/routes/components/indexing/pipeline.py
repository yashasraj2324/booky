"""Async orchestrator — the single entrypoint called from prasing.py.

Converts Chunks → Documents → hybrid chunk → text embed → image embed →
Azure Search upload.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Sequence

from .chunking import chunk_to_document, hybrid_chunk
from .text_embeddings import embed_texts_batched, make_text_embedder
from .image_embeddings import embed_images, make_image_embedder, OpenRouterImageEmbedder
from .search_store import build_search_payload, ensure_search_index, _upload_to_search

logger = logging.getLogger(__name__)


async def index_chunks_to_azure(
    chunks: Sequence[Any],
    *,
    notebook_id: str = "",
    dry_run: bool = False,
    text_embedder=None,
    image_embedder: OpenRouterImageEmbedder | None = None,
) -> dict[str, Any]:
    """
    Main entrypoint.

    Parameters
    ----------
    chunks
        Iterable of ``Chunk`` objects from ``prasing.py``.
    notebook_id
        MongoDB ObjectId of the notebook these chunks belong to.
        Stored in Azure Search so retrieval can filter by notebook.
    dry_run
        Build chunks and embeddings but skip the index upload.
    text_embedder / image_embedder
        Optional injected instances (for tests).
    """
    # 1. Convert to LangChain Documents
    documents = [chunk_to_document(c) for c in chunks]

    # Inject notebook_id into every document's metadata
    if notebook_id:
        for doc in documents:
            doc.metadata["notebook_id"] = notebook_id

    logger.info("index_chunks_to_azure: %d chunks → %d documents (notebook=%s)",
                len(chunks), len(documents), notebook_id or "N/A")

    # 2. Hybrid re-chunking
    documents = hybrid_chunk(documents)

    # 3. Text embeddings (Azure OpenAI)
    texts = [d.page_content for d in documents]
    text_vectors = await embed_texts_batched(texts, embedder=text_embedder)
    logger.info("index_chunks_to_azure: embedded %d text vectors", len(text_vectors))

    # 4. Image embeddings (OpenRouter — NVIDIA Nemotron Embed VL)
    image_vectors = await embed_images(documents, image_embedder=image_embedder)
    logger.info("index_chunks_to_azure: embedded %d image vectors", len(image_vectors))

    # 5. Build upload payload
    payload = build_search_payload(documents, text_vectors, image_vectors)

    # 6. Upload
    uploaded = 0
    if dry_run:
        logger.info("index_chunks_to_azure: DRY RUN — skipping upload (%d docs)", len(payload))
    else:
        await asyncio.to_thread(ensure_search_index)
        await asyncio.to_thread(_upload_to_search, payload)
        uploaded = len(payload)

    return {
        "total_chunks": len(chunks),
        "total_documents": len(documents),
        "text_vectors": len(text_vectors),
        "image_vectors": len(image_vectors),
        "uploaded": uploaded,
        "dry_run": dry_run,
    }


def get_vectorstore():
    """
    Return a LangChain ``AzureSearch`` vectorstore for retrieval / RAG.

    Example::

        vs = get_vectorstore()
        docs = vs.similarity_search(query="revenue by quarter", k=5)
    """
    from langchain_community.vectorstores import AzureSearch
    from langchain_openai import AzureOpenAIEmbeddings

    from .config import get_settings

    s = get_settings()
    embedder = AzureOpenAIEmbeddings(
        azure_endpoint=s.azure_openai_endpoint,
        api_key=s.openai_api_key_str,
        azure_deployment=s.azure_openai_embedding_deployment,
        api_version=s.azure_openai_api_version,
    )
    return AzureSearch(
        azure_search_endpoint=s.azure_search_endpoint,
        azure_search_key=s.search_admin_key_str,
        index_name=s.azure_search_index_name,
        embedding_function=embedder.embed_query,
    )
