"""Image embeddings via OpenRouter (NVIDIA Llama Nemotron Embed VL 1B v2, 2048 dims).

Images are retrieved from MongoDB GridFS (``gridfs://<file_id>`` URIs)
using the project's existing ``AsyncGridFSBucket``, encoded as base64
data URLs, and sent to OpenRouter's OpenAI-compatible /embeddings endpoint.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import Any

import requests

from .config import get_settings
from .chunking import deterministic_id  # noqa: F401 (re-exported for convenience)

logger = logging.getLogger(__name__)


class OpenRouterImageEmbedder:
    """
    Client for OpenRouter's multimodal embedding API.

    Endpoint: POST https://openrouter.ai/api/v1/embeddings
    Model:    nvidia/llama-nemotron-embed-vl-1b-v2:free

    Supports text, image, and image+text combined inputs — all produce
    2048-dim vectors in the same embedding space (cross-modal retrieval).
    """

    def __init__(self) -> None:
        s = get_settings()
        self.base_url = s.openrouter_base_url.rstrip("/")
        self.api_key = s.openrouter_api_key_str
        self.model = s.openrouter_image_embed_model
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    # -- GridFS retrieval --------------------------------------------------

    @staticmethod
    async def _read_gridfs_asset(asset_uri: str) -> bytes:
        """Read image bytes from ``gridfs://<file_id>`` asynchronously."""
        if not asset_uri.startswith("gridfs://"):
            raise ValueError(f"Unsupported asset URI scheme: {asset_uri}")

        file_id_str = asset_uri[len("gridfs://"):]
        from bson import ObjectId

        try:
            file_id = ObjectId(file_id_str)
        except Exception:
            file_id = file_id_str

        from app.database.mongodb import fs

        buf = io.BytesIO()
        await fs.download_to_stream(file_id, buf)
        return buf.getvalue()

    # -- Base64 encoding ---------------------------------------------------

    @staticmethod
    def _encode_image_b64(image_bytes: bytes) -> str:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    # -- Embedding methods -------------------------------------------------

    def embed_image_bytes(self, image_bytes: bytes) -> list[float]:
        """Embed raw image bytes via OpenRouter (synchronous HTTP call)."""
        data_url = self._encode_image_b64(image_bytes)
        payload = {
            "model": self.model,
            "input": [{"content": [{"type": "image_url",
                                     "image_url": {"url": data_url}}]}],
            "encoding_format": "float",
        }
        resp = self._session.post(
            f"{self.base_url}/embeddings", json=payload, timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("data"):
            raise ValueError("OpenRouter returned no embedding data")
        return data["data"][0]["embedding"]

    async def embed_image_from_gridfs(self, asset_uri: str) -> list[float]:
        """Retrieve from GridFS (async) then embed via OpenRouter (in thread)."""
        image_bytes = await self._read_gridfs_asset(asset_uri)
        return await asyncio.to_thread(self.embed_image_bytes, image_bytes)

    def embed_image_with_text(
        self, image_bytes: bytes, text: str
    ) -> list[float]:
        """Combined image+text embedding (synchronous)."""
        data_url = self._encode_image_b64(image_bytes)
        payload = {
            "model": self.model,
            "input": [{"content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": text},
            ]}],
            "encoding_format": "float",
        }
        resp = self._session.post(
            f"{self.base_url}/embeddings", json=payload, timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("data"):
            raise ValueError("OpenRouter returned no embedding data")
        return data["data"][0]["embedding"]


# ──────────────────────────────────────────────────────────────────────────────

def make_image_embedder() -> OpenRouterImageEmbedder | None:
    """Lazily create the embedder.  Returns None if not configured."""
    try:
        return OpenRouterImageEmbedder()
    except Exception as exc:
        logger.warning("OpenRouter not configured — skipping image embeddings: %s", exc)
        return None


async def embed_images(
    documents: list,
    image_embedder: OpenRouterImageEmbedder | None = None,
) -> dict[str, list[float]]:
    """
    Embed image/table elements that have a ``gridfs://`` asset URI.

    Returns a mapping of ``deterministic_id`` → image vector.  Documents
    without an image are omitted (their ``image_vector`` will be empty).
    """
    from langchain_core.documents import Document as _Doc  # noqa

    image_docs = [
        d for d in documents
        if (d.metadata if isinstance(d, _Doc) else {}).get("image_path", "").startswith("gridfs://")
    ]
    if not image_docs:
        return {}

    embedder = image_embedder or make_image_embedder()
    if embedder is None:
        return {}

    from .config import deterministic_id

    settings = get_settings()
    semaphore = asyncio.Semaphore(settings.image_embed_batch_size)
    result: dict[str, list[float]] = {}

    async def _embed_one(doc: _Doc) -> tuple[str, list[float] | None]:
        asset_uri = doc.metadata["image_path"]
        doc_id = deterministic_id(
            source=doc.metadata.get("source_doc_id", doc.metadata.get("source", "")),
            page=doc.metadata.get("page_number", 0),
            chunk_index=doc.metadata.get("chunk_index", 0),
            content=doc.page_content,
        )
        try:
            async with semaphore:
                vec = await embedder.embed_image_from_gridfs(asset_uri)
            logger.debug("embed_images: embedded %s (dim=%d)", asset_uri, len(vec))
            return doc_id, vec
        except Exception as exc:
            logger.warning("embed_images: failed for %s: %s", asset_uri, exc)
            return doc_id, None

    tasks = [_embed_one(doc) for doc in image_docs]
    for doc_id, vec in await asyncio.gather(*tasks):
        if vec is not None:
            result[doc_id] = vec
    return result
