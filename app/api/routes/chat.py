"""FastAPI route — NotebookLM-style chat: retrieve → rerank → grounded answer.

    POST /notebooks/{notebook_id}/chat
    Body: { query, top_k?, top_n?, use_reranker? }
    Response: { answer, citations, results, total }
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from bson import ObjectId
from bson.errors import InvalidId

from app.database.mongodb import notebooks_collection
from app.api.routes.components.security import get_current_user
from app.api.schemas.chat import ChatRequest, ChatResponse, Citation, RetrievalHit
from app.api.routes.components.indexing import retrieve_for_notebook
from app.api.routes.components.indexing.answerer import generate_grounded_answer

router = APIRouter(
    prefix="/notebooks/{notebook_id}/chat",
    tags=["Chat"],
    dependencies=[Depends(get_current_user)],
)


@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_with_notebook(notebook_id: str, body: ChatRequest):
    """
    Answer a question grounded in a notebook's indexed sources.

    Pipeline:
        1. Vector search in Azure AI Search (filtered by notebook_id)
        2. NVIDIA NIM cross-encoder reranking
        3. GPT-4o generates a grounded answer with [n] citations
    """
    try:
        obj_id = ObjectId(notebook_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid notebook ID format")

    notebook = await notebooks_collection.find_one({"_id": obj_id})
    if notebook is None:
        raise HTTPException(status_code=404, detail="Notebook not found")

    # Stage 1+2: Retrieve and rerank
    results = await retrieve_for_notebook(
        query=body.query,
        notebook_id=notebook_id,
        top_k=body.top_k,
        top_n=body.top_n,
    )

    # Stage 3: Generate grounded answer with citations
    answer_result = await generate_grounded_answer(body.query, results)

    # Build response
    hits = [
        RetrievalHit(
            content=r.content,
            score=r.score,
            source_doc_id=r.source_doc_id,
            page_number=r.page_number,
            modality=r.modality,
            element_type=r.element_type,
            chunk_id=r.chunk_id,
            image_path=r.image_path or None,
        )
        for r in results
    ]

    citations = [
        Citation(
            index=idx,
            source_doc_id=results[idx - 1].source_doc_id if 1 <= idx <= len(results) else "",
            page_number=results[idx - 1].page_number if 1 <= idx <= len(results) else 0,
            chunk_id=results[idx - 1].chunk_id if 1 <= idx <= len(results) else "",
            modality=results[idx - 1].modality if 1 <= idx <= len(results) else "text",
        )
        for idx in answer_result["cited_indices"]
    ]

    return ChatResponse(
        query=body.query,
        notebook_id=notebook_id,
        answer=answer_result["answer"],
        citations=citations,
        results=hits,
        total=len(hits),
    )
