"""Azure AI Search — index creation and document upload (mergeOrUpload)."""

from __future__ import annotations

import logging
from typing import Any

from .config import (
    IMAGE_EMBEDDING_DIM,
    TEXT_EMBEDDING_DIM,
    deterministic_id,
    get_settings,
)

logger = logging.getLogger(__name__)


def ensure_search_index() -> None:
    """Create the Azure AI Search index if it doesn't already exist (idempotent)."""
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents.indexes import SearchIndexClient

    s = get_settings()
    client = SearchIndexClient(
        s.azure_search_endpoint,
        AzureKeyCredential(s.search_admin_key_str),
    )

    existing = {x.name for x in client.list_indexes()}
    if s.azure_search_index_name in existing:
        logger.info("ensure_search_index: '%s' already exists", s.azure_search_index_name)
        return

    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        HnswParameters,
        SearchableField,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SemanticConfiguration,
        SemanticField,
        SemanticPrioritizedFields,
        SimpleField,
        VectorSearch,
        VectorSearchProfile,
        VectorSearchAlgorithmMetric,
    )

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=TEXT_EMBEDDING_DIM,
            vector_search_profile_name="hnsw-profile",
        ),
        SearchField(
            name="image_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=IMAGE_EMBEDDING_DIM,
            vector_search_profile_name="hnsw-profile",
        ),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="source_doc_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="page_number", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SimpleField(name="element_type", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="modality", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="content_hash", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="image_path", type=SearchFieldDataType.String),
        SimpleField(name="chunk_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-algo",
                parameters=HnswParameters(
                    metric=VectorSearchAlgorithmMetric.COSINE,
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                ),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="hnsw-profile",
                algorithm_configuration_name="hnsw-algo",
            )
        ],
    )

    semantic_config = SemanticConfiguration(
        name="semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            prioritized_content_fields=[SemanticField(field_name="content")],
        ),
    )

    index = SearchIndex(
        name=s.azure_search_index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search={"configurations": [semantic_config]},
    )

    client.create_index(index)
    logger.info("ensure_search_index: created '%s'", s.azure_search_index_name)


def build_search_payload(
    documents: list,
    text_vectors: list[list[float]],
    image_vectors: dict[str, list[float]],
) -> list[dict[str, Any]]:
    """Build Azure Search document dicts with @search.action=mergeOrUpload."""
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
            "@search.action": "mergeOrUpload",
            "id": doc_id,
            "content": doc.page_content,
            "content_vector": tvec,
            "image_vector": image_vectors.get(doc_id, []),
            "source": meta.get("source", ""),
            "source_doc_id": meta.get("source_doc_id", ""),
            "page_number": meta.get("page_number", 0),
            "element_type": meta.get("element_type", "Text"),
            "modality": meta.get("modality", "text"),
            "content_hash": meta.get("content_hash", ""),
            "image_path": meta.get("image_path", ""),
            "chunk_id": meta.get("chunk_id", ""),
            "chunk_index": meta.get("chunk_index", 0),
        })
    return payload


def _upload_to_search(payload: list[dict[str, Any]]) -> None:
    """Synchronous upload — wrapped in asyncio.to_thread by the caller."""
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient

    s = get_settings()
    client = SearchClient(
        s.azure_search_endpoint,
        s.azure_search_index_name,
        AzureKeyCredential(s.search_admin_key_str),
    )

    batch_size = s.index_batch_size
    for start in range(0, len(payload), batch_size):
        batch = payload[start : start + batch_size]
        result = client.merge_or_upload_documents(batch)
        succeeded = sum(1 for r in result if r.succeeded)
        logger.info(
            "_upload_to_search: batch %d-%d → %d succeeded, %d failed",
            start, start + len(batch), succeeded, len(batch) - succeeded,
        )
