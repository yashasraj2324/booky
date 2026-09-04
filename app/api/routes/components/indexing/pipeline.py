"""Async orchestrator — chunk → gateway embeddings → Cosmos DB upsert.

Replaces the previous Azure AI Search pipeline.  All embeddings route
through the Pydantic gateway; vectors are stored in Cosmos DB.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import Any, Sequence

from .chunking import chunk_to_document, hybrid_chunk
from .gateway import GatewayClient, get_gateway, embed_texts_batched
from .vector_store import (
    build_cosmos_documents,
    ensure_vector_index,
    upsert_chunks,
)

logger = logging.getLogger(__name__)


async def _embed_images_from_gridfs(
    documents: list,
    gateway: GatewayClient,
) -> dict[str, list[float]]:
    """
    Embed image/table elements that have a ``gridfs://`` asset URI.

    Retrieves image bytes from GridFS (async), then sends to the gateway
    for multimodal embedding (synchronous HTTP → thread).
    """
    from langchain_core.documents import Document as _Doc
    from .config import deterministic_id

    image_docs = [
        d for d in documents
        if isinstance(d, _Doc)
        and d.metadata.get("image_path", "").startswith("gridfs://")
    ]
    if not image_docs:
        return {}

    from app.database.mongodb import fs
    from .config import get_settings

    settings = get_settings()
    semaphore = asyncio.Semaphore(settings.image_embed_batch_size)
    result: dict[str, list[float]] = {}

    async def _embed_one(doc: _Doc) -> tuple[str, list[float] | None]:
        asset_uri = doc.metadata["image_path"]
        doc_id = deterministic_id(
            source=doc.metadata.get("source_doc_id", doc.metadata.get("source", "")),
            page=doc.metadata.get("page_number", 0),
            chunk_index=doc.metadata.get("chunk_index", 0),
            content=doc.page_content,
        )
        try:
            # Read from GridFS
            file_id_str = asset_uri[len("gridfs://"):]
            from bson import ObjectId
            try:
                file_id = ObjectId(file_id_str)
            except Exception:
                file_id = file_id_str

            buf = io.BytesIO()
            async with semaphore:
                await fs.download_to_stream(file_id, buf)
                # Embed via gateway (sync HTTP → thread)
                vec = await asyncio.to_thread(
                    gateway.embed_image_bytes, buf.getvalue()
                )
            logger.debug("embed_images: embedded %s (dim=%d)", asset_uri, len(vec))
            return doc_id, vec
        except Exception as exc:
            logger.warning("embed_images: failed for %s: %s", asset_uri, exc)
            return doc_id, None

    tasks = [_embed_one(doc) for doc in image_docs]
    for doc_id, vec in await asyncio.gather(*tasks):
        if vec is not None:
            result[doc_id] = vec
    return result


async def index_chunks_to_cosmos(
    chunks: Sequence[Any],
    *,
    notebook_id: str = "",
    dry_run: bool = False,
    gateway: GatewayClient | None = None,
) -> dict[str, Any]:
    """
    Main entrypoint — convert pipeline chunks to LangChain Documents,
    re-split oversized chunks, embed text + images via the gateway,
    and upsert into Cosmos DB.

    Parameters
    ----------
    chunks
        Iterable of ``Chunk`` objects from ``prasing.py``.
    notebook_id
        MongoDB ObjectId of the notebook these chunks belong to.
    dry_run
        Build chunks and embeddings but skip the Cosmos DB upsert.
    gateway
        Optional injected ``GatewayClient`` instance (for tests).
    """
    gw = gateway or get_gateway()

    # 1. Convert to LangChain Documents
    documents = [chunk_to_document(c) for c in chunks]

    if notebook_id:
        for doc in documents:
            doc.metadata["notebook_id"] = notebook_id

    logger.info(
        "index_chunks_to_cosmos: %d chunks → %d documents (notebook=%s)",
        len(chunks), len(documents), notebook_id or "N/A",
    )

    # 2. Hybrid re-chunking
    documents = hybrid_chunk(documents)

    # 3. Text embeddings (via gateway)
    texts = [d.page_content for d in documents]
    text_vectors = await embed_texts_batched(texts, gateway=gw)
    logger.info("index_chunks_to_cosmos: embedded %d text vectors", len(text_vectors))

    # 4. Image embeddings (via gateway, from GridFS)
    image_vectors = await _embed_images_from_gridfs(documents, gw)
    logger.info("index_chunks_to_cosmos: embedded %d image vectors", len(image_vectors))

    # 5. Build Cosmos DB documents
    payload = build_cosmos_documents(documents, text_vectors, image_vectors)

    # 6. Upsert
    uploaded = 0
    if dry_run:
        logger.info(
            "index_chunks_to_cosmos: DRY RUN — skipping upsert (%d docs)",
            len(payload),
        )
    else:
        await ensure_vector_index()
        uploaded = await upsert_chunks(payload)

    return {
        "total_chunks": len(chunks),
        "total_documents": len(documents),
        "text_vectors": len(text_vectors),
        "image_vectors": len(image_vectors),
        "uploaded": uploaded,
        "dry_run": dry_run,
    }
