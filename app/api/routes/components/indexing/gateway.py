"""Pydantic gateway client — unified OpenAI-compatible entrypoint.

All AI calls (text embeddings, image embeddings, chat completions, reranking)
go through this single gateway.  The gateway routes to different providers
based on the ``model`` field in the request body.

The gateway exposes:
    POST /v1/embeddings      — text + image embeddings (OpenAI format)
    POST /v1/chat/completions — chat completions (OpenAI format)
    POST /v1/ranking         — reranking (NVIDIA NIM format)

This replaces the previous direct AzureOpenAIEmbeddings, OpenRouter, and
NVIDIARerank clients.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

import requests

from .config import get_settings

logger = logging.getLogger(__name__)


class GatewayClient:
    """
    Synchronous HTTP client for the Pydantic gateway.

    All methods are synchronous because they use ``requests``.  When called
    from async code, wrap them in ``asyncio.to_thread(...)``.
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
        """
        Embed a list of texts via POST /v1/embeddings.

        Returns one vector per input text, in order.
        """
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
        # OpenAI response: {"data": [{"embedding": [...], "index": 0}, ...]}
        return [d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_texts([query])[0]

    # ── Image embeddings (multimodal) ─────────────────────────────────

    def embed_image_bytes(self, image_bytes: bytes) -> list[float]:
        """
        Embed raw image bytes via POST /v1/embeddings with multimodal content.

        The image is encoded as a base64 data URL and sent in the OpenRouter /
        OpenAI multimodal content array format.
        """
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

    # ── Chat completions (grounded answers) ───────────────────────────

    def chat_completions(self, messages: list[dict[str, str]]) -> str:
        """
        Generate a chat completion via POST /v1/chat/completions.

        Returns the assistant's message content as a string.
        """
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

    # ── Reranking (NVIDIA NIM format) ─────────────────────────────────

    def rerank(
        self,
        query: str,
        passages: list[str],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank passages by relevance to query via POST /v1/ranking.

        Returns list of ``{"index": int, "relevance_score": float}`` dicts,
        sorted by relevance (most relevant first).
        """
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
        # NVIDIA NIM response: {"rankings": [{"index": int, "logit": float}, ...]}
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
    """Cached singleton — import this, not GatewayClient directly."""
    global _client
    if _client is None:
        _client = GatewayClient()
    return _client


# ──────────────────────────────────────────────────────────────────────────────
# Async wrappers (for use in the pipeline)
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
