"""Config package — re-exports everything from config.py."""

from .config import (
    IndexingSettings,
    get_settings,
    deterministic_id,
    TEXT_EMBEDDING_DIM,
    IMAGE_EMBEDDING_DIM,
    GatewayClient,
    get_gateway,
    embed_texts_batched,
    embed_query_async,
)

__all__ = [
    "IndexingSettings",
    "get_settings",
    "deterministic_id",
    "TEXT_EMBEDDING_DIM",
    "IMAGE_EMBEDDING_DIM",
    "GatewayClient",
    "get_gateway",
    "embed_texts_batched",
    "embed_query_async",
]
