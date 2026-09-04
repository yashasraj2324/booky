"""Grounded answer generation — NotebookLM-style "answer from your sources".

Takes the reranked retrieval results, builds a prompt with numbered source
passages, sends to the Pydantic gateway's /v1/chat/completions endpoint
(routed to GPT-4o via the model field), and returns an answer with inline
[n] citations.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from .config import get_settings, get_gateway
from .retrieval import RetrievalResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a precise research assistant. Answer the user's question using ONLY \
the information in the provided source passages below.

Rules:
1. Answer in clear, natural prose. Be concise but complete.
2. Cite sources using [1], [2], [3] etc. — the number in square brackets \
   matches the passage number. Place the citation at the END of the sentence \
   it supports. You may combine citations like [1][3].
3. If multiple passages support a claim, cite all of them.
4. If the passages do NOT contain enough information to answer, say: \
   "I couldn't find enough information in the sources to answer this."
5. Do NOT use any outside knowledge. Only what is in the passages.
6. If the question is about a chart, diagram, or image, use the caption/OCR \
   text provided for that image as your source.

Source passages:
"""

USER_PROMPT_TEMPLATE = "Question: {query}"


def _build_messages(
    query: str,
    results: list[RetrievalResult],
) -> list[dict[str, str]]:
    """Build chat messages with numbered source passages."""
    passages = []
    for i, r in enumerate(results, 1):
        label = f"[{i}] "
        if r.source_doc_id:
            label += f"Source: {r.source_doc_id}, "
        if r.page_number:
            label += f"Page: {r.page_number}, "
        label += f"Type: {r.modality}"
        passages.append(f"{label}\n{r.content}")

    system_content = SYSTEM_PROMPT + "\n\n" + "\n\n---\n\n".join(passages)
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(query=query)},
    ]


def _extract_citation_indices(answer: str) -> list[int]:
    """Extract all unique citation numbers [1], [2], etc. from the answer."""
    matches = re.findall(r"\[(\d+)\]", answer)
    return sorted(set(int(m) for m in matches))


async def generate_grounded_answer(
    query: str,
    results: list[RetrievalResult],
    gateway=None,
) -> dict[str, Any]:
    """
    Generate a grounded answer from retrieved chunks via the gateway.

    Returns dict with:
        - answer: str — LLM answer with [n] citations
        - cited_indices: list[int] — passage numbers cited
        - cited_results: list[RetrievalResult] — the actual chunks cited
    """
    if not results:
        return {
            "answer": "I couldn't find enough information in the sources to answer this.",
            "cited_indices": [],
            "cited_results": [],
        }

    gw = gateway or get_gateway()
    messages = _build_messages(query, results)

    # Gateway chat completions is synchronous → offload to thread
    answer_text = await asyncio.to_thread(gw.chat_completions, messages)

    cited_indices = _extract_citation_indices(answer_text)
    cited_results = [results[i - 1] for i in cited_indices if 1 <= i <= len(results)]

    logger.info(
        "generate_grounded_answer: %d results → answer with %d citations",
        len(results), len(cited_indices),
    )

    return {
        "answer": answer_text,
        "cited_indices": cited_indices,
        "cited_results": cited_results,
    }
