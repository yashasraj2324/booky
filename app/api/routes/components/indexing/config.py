"""Configuration for Azure OpenAI, OpenRouter, and Azure AI Search."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Dimension constants — must match the deployed models
TEXT_EMBEDDING_DIM = 3072   # Azure OpenAI text-embedding-3-large
IMAGE_EMBEDDING_DIM = 2048  # NVIDIA Llama Nemotron Embed VL 1B v2 (via OpenRouter)


class IndexingSettings(BaseSettings):
    """Validated configuration loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Azure OpenAI (text embeddings) ───────────────────────────────
    azure_openai_endpoint: str = Field(
        ..., description="AOAI resource endpoint"
    )
    azure_openai_api_key: SecretStr = Field(
        ..., description="AOAI resource key"
    )
    azure_openai_api_version: str = Field(
        default="2024-02-01",
    )
    azure_openai_embedding_deployment: str = Field(
        ..., description="Deployment name for text-embedding-3-large"
    )

    # ── OpenRouter (multimodal image embeddings) ─────────────────────
    openrouter_api_key: SecretStr = Field(
        ..., description="OpenRouter API key"
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
    )
    openrouter_image_embed_model: str = Field(
        default="nvidia/llama-nemotron-embed-vl-1b-v2:free",
    )

    # ── Azure AI Search ──────────────────────────────────────────────
    azure_search_endpoint: str = Field(
        ..., description="Search service endpoint"
    )
    azure_search_admin_key: SecretStr = Field(
        ..., description="Search service admin key"
    )
    azure_search_index_name: str = Field(
        default="booky-documents",
    )

    # ── Tunables ─────────────────────────────────────────────────────
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    embed_batch_size: int = Field(default=16)
    image_embed_batch_size: int = Field(default=4)
    index_batch_size: int = Field(default=50)

    # ── Accessors ────────────────────────────────────────────────────
    @property
    def openai_api_key_str(self) -> str:
        return self.azure_openai_api_key.get_secret_value()

    @property
    def openrouter_api_key_str(self) -> str:
        return self.openrouter_api_key.get_secret_value()

    @property
    def search_admin_key_str(self) -> str:
        return self.azure_search_admin_key.get_secret_value()

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
    """Stable hash ID for Azure Search mergeOrUpload (idempotent re-runs)."""
    raw = f"{source}|{page}|{chunk_index}|{content[:200]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:64]
