"""Retrieval pipeline — vector search in Azure AI Search + NVIDIA reranker.

Syncs retrieval with the existing notebook/source model:

    Notebook (MongoDB) → has many → Sources (MongoDB)
         ↓                                      ↓
    notebook_id                          source_doc_id
         ↓                                      ↓
    Azure AI Search index (booky-documents)
    Fields: notebook_id, source_doc_id, content, content_vector, ...

Flow:
    1. User asks a question against a notebook.
    2. ``retrieve()`` queries Azure AI Search filtered by ``notebook_id``
       using the Azure OpenAI text embedding of the query.
    3. The top-K candidates (e.g. 20) are passed to the NVIDIA reranker.
    4. The reranked top-N (e.g. 5) are returned with relevance scores.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from .config import get_settings
from .reranker import rerank_documents

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieved chunk with relevance score."""
    content: str
    score: float
    source_doc_id: str
    notebook_id: str
    page_number: int
    modality: str
    element_type: str
    chunk_id: str
    image_path: str
    metadata: dict[str, Any]


def get_search_client():
    """Lazily create an Azure Search client for queries."""
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient

    s = get_settings()
    return SearchClient(
        s.azure_search_endpoint,
        s.azure_search_index_name,
        AzureKeyCredential(s.search_admin_key_str),
    )


def _raw_search(
    query_text: str,
    notebook_id: str | None = None,
    source_doc_id: str | None = None,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Perform a vector similarity search against Azure AI Search."""
    from langchain_openai import AzureOpenAIEmbeddings
    from azure.search.documents.models import VectorizedQuery

    s = get_settings()

    embedder = AzureOpenAIEmbeddings(
        azure_endpoint=s.azure_openai_endpoint,
        api_key=s.openai_api_key_str,
        azure_deployment=s.azure_openai_embedding_deployment,
        api_version=s.azure_openai_api_version,
    )
    query_vector = embedder.embed_query(query_text)

    client = get_search_client()

    filters = []
    if notebook_id:
        filters.append(f"notebook_id eq '{notebook_id}'")
    if source_doc_id:
        filters.append(f"source_doc_id eq '{source_doc_id}'")
    filter_expr = " and ".join(filters) if filters else None

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="content_vector",
    )

    results = client.search(
        search_text=None,
        vector_queries=[vector_query],
        filter=filter_expr,
        select=[
            "id", "content", "source_doc_id", "notebook_id",
            "page_number", "modality", "element_type",
            "chunk_id", "image_path", "chunk_index",
        ],
        top=top_k,
    )

    return [dict(r) for r in results]


async def retrieve(
    query: str,
    notebook_id: str | None = None,
    source_doc_id: str | None = None,
    top_k: int = 20,
    top_n: int | None = None,
    use_reranker: bool = True,
) -> list[RetrievalResult]:
    """Full retrieval pipeline: vector search → rerank."""
    raw_results = await asyncio.to_thread(
        _raw_search, query, notebook_id, source_doc_id, top_k
    )

    if not raw_results:
        logger.info("retrieve: no results from vector search")
        return []

    logger.info("retrieve: %d candidates from vector search", len(raw_results))

    from langchain_core.documents import Document

    documents = [
        Document(
            page_content=r.get("content", ""),
            metadata={
                "id": r.get("id", ""),
                "source_doc_id": r.get("source_doc_id", ""),
                "notebook_id": r.get("notebook_id", ""),
                "page_number": r.get("page_number", 0),
                "modality": r.get("modality", "text"),
                "element_type": r.get("element_type", "Text"),
                "chunk_id": r.get("chunk_id", ""),
                "image_path": r.get("image_path", ""),
                "chunk_index": r.get("chunk_index", 0),
                "similarity_score": r.get("@search.score", 0.0),
            },
        )
        for r in raw_results
    ]

    if use_reranker:
        reranked = await rerank_documents(query, documents)
    else:
        reranked = documents

    n = top_n or get_settings().rerank_top_n
    reranked = reranked[:n]

    results = [
        RetrievalResult(
            content=doc.page_content,
            score=doc.metadata.get("relevance_score",
                                   doc.metadata.get("similarity_score", 0.0)),
            source_doc_id=doc.metadata.get("source_doc_id", ""),
            notebook_id=doc.metadata.get("notebook_id", ""),
            page_number=doc.metadata.get("page_number", 0),
            modality=doc.metadata.get("modality", "text"),
            element_type=doc.metadata.get("element_type", "Text"),
            chunk_id=doc.metadata.get("chunk_id", ""),
            image_path=doc.metadata.get("image_path", ""),
            metadata=doc.metadata,
        )
        for doc in reranked
    ]

    logger.info("retrieve: returning %d results", len(results))
    return results


async def retrieve_for_notebook(
    query: str,
    notebook_id: str,
    top_k: int = 20,
    top_n: int | None = None,
) -> list[RetrievalResult]:
    """Convenience: retrieve scoped to a notebook."""
    return await retrieve(query=query, notebook_id=notebook_id, top_k=top_k, top_n=top_n)
