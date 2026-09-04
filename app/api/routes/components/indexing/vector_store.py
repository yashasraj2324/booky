"""Cosmos DB for MongoDB — vector store using vCore vector search.

Replaces Azure AI Search.  Uses the same pymongo/motor AsyncMongoClient
already configured in app.database.mongodb.

Cosmos DB vCore vector search uses the $aggregate stage
``$search`` with ``vectorSearch`` to perform HNSW cosine similarity.

Document schema in the ``chunks`` collection:
    {
        _id: str (deterministic hash),
        content: str,
        content_vector: [float],   # 3072-dim text embedding
        image_vector: [float],     # 2048-dim image embedding (empty if N/A)
        notebook_id: str,
        source_doc_id: str,
        page_number: int,
        modality: str,
        element_type: str,
        chunk_id: str,
        chunk_index: int,
        content_hash: str,
        image_path: str,           # gridfs:// URI
    }
"""

from __future__ import annotations

import logging
from typing import Any

from .config import deterministic_id, get_settings

logger = logging.getLogger(__name__)


def get_chunks_collection():
    """Return the AsyncMongoClient collection for vector chunks."""
    from app.database.mongodb import db

    return db[get_settings().cosmos_container_name]


async def ensure_vector_index() -> None:
    """
    Create the vCore vector index if it doesn't exist.

    This is idempotent — safe to call on every pipeline run.

    Cosmos DB for MongoDB supports vector search via the
    ``createIndex`` command with a ``vectorSearch`` type.
    """
    collection = get_chunks_collection()

    index_name = "vector_index"
    index_path = get_settings().cosmos_vector_index_path

    # Check if index already exists
    indexes = await collection.list_indexes().to_list(100)
    for idx in indexes:
        if idx.get("name") == index_name:
            logger.info("ensure_vector_index: '%s' already exists", index_name)
            return

    # Create the vector index (Cosmos DB vCore format)
    index_spec = {
        "name": index_name,
        "key": {index_path: "cosmosSearch"},
        "cosmosSearchOptions": {
            "kind": "vector-hnsw",
            "m": 16,
            "efConstruction": 64,
            "similarity": "COS",
            "dimensions": 3072,  # TEXT_EMBEDDING_DIM
        },
    }

    try:
        await collection.create_index([index_path], **index_spec)
        logger.info("ensure_vector_index: created '%s' on %s", index_name, index_path)
    except Exception as exc:
        # Index may already exist or Cosmos DB may not support this command
        # in the current tier — log and continue
        logger.warning("ensure_vector_index: %s", exc)


async def upsert_chunks(documents: list[dict[str, Any]]) -> int:
    """
    Upsert a batch of chunk documents into Cosmos DB.

    Uses ``update_one`` with ``upsert=True`` keyed on ``_id`` (deterministic
    hash) so re-running the pipeline is idempotent.

    Returns the number of documents upserted.
    """
    collection = get_chunks_collection()
    count = 0

    for doc in documents:
        doc_id = doc.pop("_id")
        await collection.update_one(
            {"_id": doc_id},
            {"$set": doc},
            upsert=True,
        )
        count += 1

    logger.info("upsert_chunks: %d documents upserted", count)
    return count


async def vector_search(
    query_vector: list[float],
    notebook_id: str | None = None,
    source_doc_id: str | None = None,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """
    Perform a vector similarity search against Cosmos DB.

    Uses the ``$search`` aggregation stage with ``vectorSearch``.

    Returns raw documents with a ``similarityScore`` field.
    """
    collection = get_chunks_collection()

    # Build the match filter
    match_filter: dict[str, Any] = {}
    if notebook_id:
        match_filter["notebook_id"] = notebook_id
    if source_doc_id:
        match_filter["source_doc_id"] = source_doc_id

    pipeline = [
        # If we have filters, pre-filter before vector search
        *([{"$match": match_filter}] if match_filter else []),
        # Vector search
        {
            "$search": {
                "cosmosSearch": {
                    "vector": query_vector,
                    "path": get_settings().cosmos_vector_index_path,
                    "k": top_k,
                },
                "returnStoredSource": True,
            }
        },
        # Project fields we need
        {
            "$project": {
                "content": 1,
                "source_doc_id": 1,
                "notebook_id": 1,
                "page_number": 1,
                "modality": 1,
                "element_type": 1,
                "chunk_id": 1,
                "image_path": 1,
                "chunk_index": 1,
                "similarityScore": {"$meta": "searchScore"},
            }
        },
    ]

    results = await collection.aggregate(pipeline).to_list(top_k)
    return results


def build_cosmos_documents(
    documents: list,
    text_vectors: list[list[float]],
    image_vectors: dict[str, list[float]],
) -> list[dict[str, Any]]:
    """Build Cosmos DB document dicts from LangChain Documents + vectors."""
    payload = []
    for doc, tvec in zip(documents, text_vectors):
        meta = doc.metadata
        doc_id = deterministic_id(
            source=meta.get("source_doc_id", meta.get("source", "")),
            page=meta.get("page_number", 0),
            chunk_index=meta.get("chunk_index", 0),
            content=doc.page_content,
        )
        payload.append({
            "_id": doc_id,
            "content": doc.page_content,
            "content_vector": tvec,
            "image_vector": image_vectors.get(doc_id, []),
            "notebook_id": meta.get("notebook_id", ""),
            "source_doc_id": meta.get("source_doc_id", ""),
            "source": meta.get("source", ""),
            "page_number": meta.get("page_number", 0),
            "modality": meta.get("modality", "text"),
            "element_type": meta.get("element_type", "Text"),
            "chunk_id": meta.get("chunk_id", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "content_hash": meta.get("content_hash", ""),
            "image_path": meta.get("image_path", ""),
        })
    return payload
