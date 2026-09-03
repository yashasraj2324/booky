"""Convert pipeline Chunks to LangChain Documents and hybrid re-chunk."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import get_settings

logger = logging.getLogger(__name__)


def chunk_to_document(chunk: Any) -> Document:
    """
    Convert a ``Chunk`` dataclass from prasing.py into a LangChain
    ``Document``.

    Chunk fields mapped:
        text_representation → page_content
        chunk_id            → metadata["chunk_id"]
        source_doc_id       → metadata["source_doc_id"]
        page_number         → metadata["page_number"]
        modality            → metadata["modality"]
        metadata.asset_path → metadata["image_path"]  (gridfs:// URI)
        metadata.element_type → metadata["element_type"]
        metadata.source_filename → metadata["source"]
    """
    text = getattr(chunk, "text_representation", "") or ""
    raw_meta = getattr(chunk, "metadata", {}) or {}
    meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}

    meta["chunk_id"] = getattr(chunk, "chunk_id", "")
    meta["source_doc_id"] = getattr(chunk, "source_doc_id", "")
    meta["notebook_id"] = raw_meta.get("notebook_id", "")
    meta["page_number"] = getattr(chunk, "page_number", 0) or 0
    meta["modality"] = getattr(chunk, "modality", "text")
    meta["element_type"] = raw_meta.get("element_type", "Text")
    meta["source"] = raw_meta.get("source_filename", "")
    meta.setdefault("content_hash", hashlib.sha256(text.encode()).hexdigest())
    meta.setdefault("chunk_index", 0)
    meta["image_path"] = raw_meta.get("asset_path", "") or ""

    return Document(page_content=text, metadata=meta)


def hybrid_chunk(
    documents: Sequence[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Keep structural grouping from chunk_by_title, then sub-split only the
    oversized **text** chunks via RecursiveCharacterTextSplitter.

    Image and table chunks are never split — their text_representation is
    a compact caption / OCR result that should stay whole.
    """
    settings = get_settings()
    cs = chunk_size or settings.chunk_size
    co = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cs,
        chunk_overlap=co,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    result: list[Document] = []
    global_idx = 0

    for doc in documents:
        is_visual = doc.metadata.get("modality") in ("image", "table")
        if is_visual or len(doc.page_content) <= cs:
            meta = dict(doc.metadata)
            meta["chunk_index"] = global_idx
            result.append(Document(page_content=doc.page_content, metadata=meta))
            global_idx += 1
        else:
            for sub in splitter.split_documents([doc]):
                meta = dict(sub.metadata)
                meta["chunk_index"] = global_idx
                result.append(Document(page_content=sub.page_content, metadata=meta))
                global_idx += 1

    logger.info(
        "hybrid_chunk: %d input → %d output (chunk_size=%d)",
        len(documents), len(result), cs,
    )
    return result
