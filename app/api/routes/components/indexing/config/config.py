"""Settings, HTTP client, and helpers for the indexing pipeline.

This module consolidates:
  - IndexingSettings  — pydantic-settings loaded from .env
  - GatewayClient     — OpenAI-compatible HTTP client (embeddings, chat, rerank)
  - get_gateway()      — cached singleton
  - embed_texts_batched / embed_query_async — async wrappers
  - deterministic_id   — stable hash for idempotent upserts
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
from functools import lru_cache
from typing import Any

import requests
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Dimension constants — must match the models configured behind the gateway
# ──────────────────────────────────────────────────────────────────────────────
TEXT_EMBEDDING_DIM = 3072   # text-embedding-3-large
IMAGE_EMBEDDING_DIM = 2048  # nvidia/llama-nemotron-embed-vl-1b-v2


# ──────────────────────────────────────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────────────────────────────────────

class IndexingSettings(BaseSettings):
    """Validated configuration loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Gateway (unified entrypoint for all AI calls) ──────────────────
    gateway_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="Base URL for the OpenAI-compatible gateway",
    )
    gateway_api_key: SecretStr = Field(
        default=SecretStr("dummy"),
        description="API key for the gateway",
    )

    # ── Model names (passed as ``model`` field in gateway requests) ───
    text_embedding_model: str = Field(
        default="text-embedding-3-large",
    )
    image_embedding_model: str = Field(
        default="nvidia/llama-nemotron-embed-vl-1b-v2:free",
    )
    chat_model: str = Field(
        default="gpt-4o",
    )
    rerank_model: str = Field(
        default="nvidia/nv-rerankqa-mistral-4b-v3",
    )
    chat_temperature: float = Field(
        default=0.3,
    )

    # ── Cosmos DB ───────────────────────────────────────────────────────
    cosmos_container_name: str = Field(
        default="chunks",
    )
    cosmos_vector_index_path: str = Field(
        default="/content_vector",
    )

    # ── Tunables ──────────────────────────────────────────────────────
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    embed_batch_size: int = Field(default=16)
    image_embed_batch_size: int = Field(default=4)
    rerank_top_n: int = Field(default=5)

    # ── Accessors ──────────────────────────────────────────────────────
    @property
    def gateway_api_key_str(self) -> str:
        return self.gateway_api_key.get_secret_value()

    # ── Validation ─────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# HTTP client — all AI calls go through this
# ──────────────────────────────────────────────────────────────────────────────

class GatewayClient:
    """
    Synchronous HTTP client for the OpenAI-compatible gateway.

    All methods are synchronous (uses ``requests``).  When called from
    async code, wrap them in ``asyncio.to_thread(...)``.
    """

    def __init__(self) -> None:
        s = get_settings()
        self.base_url = s.gateway_base_url.rstrip("/")
        self.api_key = s.gateway_api_key_str
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    # ── Text embeddings ──────────────────────────────────────────────

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts via POST /v1/embeddings."""
        s = get_settings()
        payload = {
            "model": s.text_embedding_model,
            "input": texts,
            "encoding_format": "float",
        }
        resp = self._session.post(
            f"{self.base_url}/embeddings", json=payload, timeout=120
        )
        resp.raise_for_status()
        data = resp.json()
        return [d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_texts([query])[0]

    # ── Image embeddings (multimodal) ─────────────────────────────────

    def embed_image_bytes(self, image_bytes: bytes) -> list[float]:
        """Embed raw image bytes via POST /v1/embeddings with multimodal content."""
        s = get_settings()
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"

        payload = {
            "model": s.image_embedding_model,
            "input": [
                {
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        }
                    ]
                }
            ],
            "encoding_format": "float",
        }
        resp = self._session.post(
            f"{self.base_url}/embeddings", json=payload, timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("data"):
            raise ValueError("Gateway returned no embedding data for image")
        return data["data"][0]["embedding"]

    # ── Chat completions ─────────────────────────────────────────────

    def chat_completions(self, messages: list[dict[str, str]]) -> str:
        """Generate a chat completion via POST /v1/chat/completions."""
        s = get_settings()
        payload = {
            "model": s.chat_model,
            "messages": messages,
            "temperature": s.chat_temperature,
            "max_tokens": 1500,
        }
        resp = self._session.post(
            f"{self.base_url}/chat/completions", json=payload, timeout=120
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    # ── Reranking ────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        passages: list[str],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank passages by relevance to query via POST /v1/ranking."""
        s = get_settings()
        n = top_n or s.rerank_top_n
        payload = {
            "model": s.rerank_model,
            "query": {"text": query},
            "passages": [{"text": p} for p in passages],
            "top_n": n,
            "truncate": "END",
        }
        resp = self._session.post(
            f"{self.base_url}/ranking", json=payload, timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        rankings = data.get("rankings", data.get("data", []))
        return [
            {"index": r["index"], "relevance_score": r.get("logit", r.get("score", 0.0))}
            for r in rankings
        ]


# ──────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ──────────────────────────────────────────────────────────────────────────────

_client: GatewayClient | None = None


def get_gateway() -> GatewayClient:
    """Cached singleton."""
    global _client
    if _client is None:
        _client = GatewayClient()
    return _client


# ──────────────────────────────────────────────────────────────────────────────
# Async wrappers
# ──────────────────────────────────────────────────────────────────────────────

async def embed_texts_batched(
    texts: list[str],
    gateway: GatewayClient | None = None,
) -> list[list[float]]:
    """Embed texts in batches, offloading to a thread."""
    gw = gateway or get_gateway()
    batch_size = get_settings().embed_batch_size
    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batch_vectors = await asyncio.to_thread(gw.embed_texts, batch)
        vectors.extend(batch_vectors)
        logger.debug("embed_texts_batched: %d/%d done", len(vectors), len(texts))

    return vectors


async def embed_query_async(
    query: str,
    gateway: GatewayClient | None = None,
) -> list[float]:
    """Embed a single query string (async wrapper)."""
    gw = gateway or get_gateway()
    return await asyncio.to_thread(gw.embed_query, query)
