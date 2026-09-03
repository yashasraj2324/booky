"""Schemas for the chat/retrieval API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Query a notebook's indexed sources."""
    notebook_id: str = Field(description="MongoDB ObjectId of the notebook to query")
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=20, description="Candidates from vector search")
    top_n: int = Field(default=5, description="Final results after reranking")
    use_reranker: bool = Field(default=True)


class RetrievalHit(BaseModel):
    content: str
    score: float
    source_doc_id: str
    page_number: int
    modality: str
    element_type: str
    chunk_id: str
    image_path: str | None = None


class Citation(BaseModel):
    """A citation reference — maps [n] in the answer to a source chunk."""
    index: int = Field(description="The [n] number used in the answer text")
    source_doc_id: str
    page_number: int
    chunk_id: str
    modality: str


class ChatResponse(BaseModel):
    query: str
    notebook_id: str
    answer: str = Field(description="LLM-generated grounded answer with [n] citations")
    citations: list[Citation] = Field(default_factory=list)
    results: list[RetrievalHit] = Field(default_factory=list)
    total: int
