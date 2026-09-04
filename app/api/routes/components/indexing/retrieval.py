"""Retrieval pipeline — LangChain + LangGraph with Cosmos DB vector search.

Uses LangGraph to build a stateful retrieval graph:
    1. embed_query:   Gateway embeds the user's query
    2. vector_search:  Cosmos DB vCore vector similarity search (top_k)
    3. rerank:         NVIDIA cross-encoder reranker via gateway (top_n)
    4. (return results)

The graph is compiled once and can be invoked with a single async call.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Sequence
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from .config import get_settings, get_gateway
from .vector_store import vector_search
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


# ──────────────────────────────────────────────────────────────────────────────
# LangGraph state
# ──────────────────────────────────────────────────────────────────────────────

class RetrievalState(TypedDict, total=False):
    query: str
    notebook_id: str | None
    source_doc_id: str | None
    top_k: int
    top_n: int | None
    use_reranker: bool
    query_vector: list[float]
    raw_results: list[dict[str, Any]]
    documents: list[Document]
    reranked: list[Document]
    results: list[RetrievalResult]


# ──────────────────────────────────────────────────────────────────────────────
# Graph node functions
# ──────────────────────────────────────────────────────────────────────────────

async def _embed_query_node(state: RetrievalState) -> dict[str, Any]:
    """Embed the query string via the gateway."""
    gw = get_gateway()
    query_vector = await asyncio.to_thread(gw.embed_query, state["query"])
    return {"query_vector": query_vector}


async def _vector_search_node(state: RetrievalState) -> dict[str, Any]:
    """Perform vector similarity search against Cosmos DB."""
    raw_results = await vector_search(
        query_vector=state["query_vector"],
        notebook_id=state.get("notebook_id"),
        source_doc_id=state.get("source_doc_id"),
        top_k=state.get("top_k", 20),
    )

    # Convert to LangChain Documents
    documents = [
        Document(
            page_content=r.get("content", ""),
            metadata={
                "id": r.get("_id", ""),
                "source_doc_id": r.get("source_doc_id", ""),
                "notebook_id": r.get("notebook_id", ""),
                "page_number": r.get("page_number", 0),
                "modality": r.get("modality", "text"),
                "element_type": r.get("element_type", "Text"),
                "chunk_id": r.get("chunk_id", ""),
                "image_path": r.get("image_path", ""),
                "chunk_index": r.get("chunk_index", 0),
                "similarity_score": r.get("similarityScore", 0.0),
            },
        )
        for r in raw_results
    ]

    logger.info("vector_search_node: %d candidates", len(documents))
    return {"raw_results": raw_results, "documents": documents}


async def _rerank_node(state: RetrievalState) -> dict[str, Any]:
    """Rerank documents via NVIDIA NIM through the gateway."""
    documents = state.get("documents", [])
    if not documents:
        return {"reranked": [], "results": []}

    reranked = await rerank_documents(state["query"], documents)

    n = state.get("top_n") or get_settings().rerank_top_n
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

    return {"reranked": reranked, "results": results}


def _should_rerank(state: RetrievalState) -> str:
    """Conditional edge: skip reranker if disabled or no results."""
    if not state.get("use_reranker", True):
        return "skip"
    if not state.get("documents"):
        return "skip"
    return "rerank"


async def _skip_rerank_node(state: RetrievalState) -> dict[str, Any]:
    """Skip reranking — convert documents directly to RetrievalResults."""
    documents = state.get("documents", [])
    n = state.get("top_n") or get_settings().rerank_top_n
    documents = documents[:n]

    results = [
        RetrievalResult(
            content=doc.page_content,
            score=doc.metadata.get("similarity_score", 0.0),
            source_doc_id=doc.metadata.get("source_doc_id", ""),
            notebook_id=doc.metadata.get("notebook_id", ""),
            page_number=doc.metadata.get("page_number", 0),
            modality=doc.metadata.get("modality", "text"),
            element_type=doc.metadata.get("element_type", "Text"),
            chunk_id=doc.metadata.get("chunk_id", ""),
            image_path=doc.metadata.get("image_path", ""),
            metadata=doc.metadata,
        )
        for doc in documents
    ]
    return {"results": results}


# ──────────────────────────────────────────────────────────────────────────────
# Build the LangGraph
# ──────────────────────────────────────────────────────────────────────────────

def _build_retrieval_graph():
    """Build and compile the LangGraph retrieval pipeline."""
    graph = StateGraph(RetrievalState)

    graph.add_node("embed_query", _embed_query_node)
    graph.add_node("vector_search", _vector_search_node)
    graph.add_node("rerank", _rerank_node)
    graph.add_node("skip_rerank", _skip_rerank_node)

    graph.set_entry_point("embed_query")
    graph.add_edge("embed_query", "vector_search")
    graph.add_conditional_edges(
        "vector_search",
        _should_rerank,
        {
            "rerank": "rerank",
            "skip": "skip_rerank",
        },
    )
    graph.add_edge("rerank", END)
    graph.add_edge("skip_rerank", END)

    return graph.compile()


# Compile once, reuse on every query
_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_retrieval_graph()
    return _compiled_graph


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

async def retrieve(
    query: str,
    notebook_id: str | None = None,
    source_doc_id: str | None = None,
    top_k: int = 20,
    top_n: int | None = None,
    use_reranker: bool = True,
) -> list[RetrievalResult]:
    """
    Full retrieval pipeline via LangGraph:
        embed query → Cosmos DB vector search → NVIDIA rerank → results
    """
    graph = _get_graph()

    initial_state: RetrievalState = {
        "query": query,
        "notebook_id": notebook_id,
        "source_doc_id": source_doc_id,
        "top_k": top_k,
        "top_n": top_n,
        "use_reranker": use_reranker,
    }

    final_state = await graph.ainvoke(initial_state)
    results: list[RetrievalResult] = final_state.get("results", [])

    logger.info("retrieve: returning %d results", len(results))
    return results


async def retrieve_for_notebook(
    query: str,
    notebook_id: str,
    top_k: int = 20,
    top_n: int | None = None,
) -> list[RetrievalResult]:
    """Convenience: retrieve scoped to a notebook."""
    return await retrieve(
        query=query, notebook_id=notebook_id, top_k=top_k, top_n=top_n
    )
