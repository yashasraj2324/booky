"""Configuration for the Pydantic gateway (embeddings + LLM) and Cosmos DB.

All embedding models (text + image) and the chat LLM are accessed through a
single OpenAI-compatible Pydantic gateway.  The ``model`` field in each
request body routes to the correct provider (NVIDIA, Azure OpenAI, etc.).

Vector storage uses Azure Cosmos DB for MongoDB with vCore vector search —
the same pymongo/motor driver already used in the project.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Dimension constants — must match the models configured in the gateway
TEXT_EMBEDDING_DIM = 3072   # text-embedding-3-large (or whatever the gateway routes to)
IMAGE_EMBEDDING_DIM = 2048  # nvidia/llama-nemotron-embed-vl-1b-v2


class IndexingSettings(BaseSettings):
    """Validated configuration loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Pydantic Gateway (unified entrypoint for all AI calls) ─────────
    gateway_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="Base URL for the Pydantic gateway (OpenAI-compatible)",
    )
    gateway_api_key: SecretStr = Field(
        default=SecretStr("dummy"),
        description="API key for the Pydantic gateway",
    )

    # ── Model names (passed as ``model`` field in gateway requests) ───
    text_embedding_model: str = Field(
        default="text-embedding-3-large",
        description="Model name for text embeddings (routed by gateway)",
    )
    image_embedding_model: str = Field(
        default="nvidia/llama-nemotron-embed-vl-1b-v2:free",
        description="Model name for multimodal image embeddings",
    )
    chat_model: str = Field(
        default="gpt-4o",
        description="Model name for chat completions (grounded answers)",
    )
    rerank_model: str = Field(
        default="nvidia/nv-rerankqa-mistral-4b-v3",
        description="Model name for NVIDIA cross-encoder reranking",
    )
    chat_temperature: float = Field(
        default=0.3,
        description="Temperature for grounded answers (low = factual)",
    )

    # ── Cosmos DB (vector store — uses existing MongoDB driver) ────────
    cosmos_container_name: str = Field(
        default="chunks",
        description="Cosmos DB container/collection name for vector search",
    )
    cosmos_vector_index_path: str = Field(
        default="/content_vector",
        description="Document path for the vector index",
    )

    # ── Tunables ─────────────────────────────────────────────────────
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    embed_batch_size: int = Field(default=16)
    image_embed_batch_size: int = Field(default=4)
    rerank_top_n: int = Field(
        default=5,
        description="Number of top documents to keep after reranking",
    )

    # ── Accessors ────────────────────────────────────────────────────
    @property
    def gateway_api_key_str(self) -> str:
        return self.gateway_api_key.get_secret_value()

    # ── Validation ───────────────────────────────────────────────────
    @field_validator("chunk_size")
    @classmethod
    def _chunk_size_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("chunk_size must be positive")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_lt_chunk(cls, v: int, info) -> int:
        cs = info.data.get("chunk_size", 1000)
        if v >= cs:
            raise ValueError(f"chunk_overlap ({v}) must be < chunk_size ({cs})")
        return v


@lru_cache(maxsize=1)
def get_settings() -> IndexingSettings:
    """Cached singleton."""
    return IndexingSettings()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def deterministic_id(source: str, page: int, chunk_index: int, content: str) -> str:
    """Stable hash ID for Cosmos DB upserts (idempotent re-runs)."""
    raw = f"{source}|{page}|{chunk_index}|{content[:200]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:64]
