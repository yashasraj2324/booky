"""
Multimodal RAG parsing pipeline using Unstructured, RapidOCR, and an
optional NVIDIA-hosted VLM for captioning.

Cost policy this pipeline implements:
  - Digital text            -> native extraction only.        No VLM.
  - Scanned page             -> OCR only.                       No VLM.
  - Table                    -> HTML extraction, OCR fallback.  Usually no VLM.
  - Simple image (photo)     -> OCR + metadata only.            No VLM.
  - Chart / diagram          -> OCR + compact VLM caption once at ingest.
  - User asks visual detail  -> VLM caption of the stored crop, on demand
                                 (see MultimodalDocumentParser.caption_on_demand).

VLM captions are:
  - opt-in (caption_images=False disables all automatic captioning)
  - restricted to visual_type in {"chart", "diagram"} by default, since
    that's where a caption earns its cost — plain photos rarely need one
  - cached by image content hash, so re-ingesting a document (or two
    documents sharing an identical figure) never pays for the same
    caption twice
  - run on a resized copy of the image (max_image_dimension), since VLM
    cost scales with image size/tokens and OCR/embedding rarely benefit
    from full resolution

Install:
  pip install "unstructured[pdf]" rapidocr-onnxruntime openai --break-system-packages
  # System dep still required: poppler-utils, libmagic (NOT tesseract-ocr)
  export NVIDIA_API_KEY=nvapi-...
  export NVIDIA_VLM_MODEL=meta/llama-3.2-90b-vision-instruct
"""

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage
from rapidocr_onnxruntime import RapidOCR

from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element, Image, Table
from unstructured.partition.auto import partition

from app.api.routes.components.indexing import index_chunks_to_cosmos

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    source_doc_id: str
    page_number: int | None
    modality: str  # "text" | "table" | "image"
    text_representation: str
    raw_asset_b64: str | None = None  # intentionally left None on persisted
    metadata: dict[str, Any] = field(default_factory=dict)  # chunks — see AssetStore


@dataclass
class ProcessingIssue:
    source_doc_id: str
    element_index: int
    stage: str
    severity: str  # "warning" | "error"
    message: str
    page_number: int | None = None


# --------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------

class OCREngine(ABC):
    @abstractmethod
    def extract_text(self, image_bytes: bytes) -> str:
        ...


class RapidOCREngine(OCREngine):
    def __init__(self) -> None:
        # Loaded once and reused — RapidOCR's ONNX models are cheap to
        # keep warm, expensive to reinitialize per call.
        self._engine = RapidOCR()

    def extract_text(self, image_bytes: bytes) -> str:
        if not image_bytes:
            return ""
        pil_img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        result, _elapsed = self._engine(np.array(pil_img))
        if not result:
            return ""
        return "\n".join(line[1] for line in result)


# --------------------------------------------------------------------------
# VLM captioning
# --------------------------------------------------------------------------

class ImageCaptioner(ABC):
    model_name: str = "unknown"

    @abstractmethod
    def caption(self, image_bytes: bytes) -> str:
        ...


class NvidiaVLMCaptioner(ImageCaptioner):
    """Captioner backed by an NVIDIA-hosted VLM (build.nvidia.com)."""

    PROMPT = (
        "Describe this image/figure/chart in detail for search retrieval. "
        "Include any axis labels, trends, numbers, or key visual elements. "
        "Be concise — 2-4 sentences."
    )

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        from openai import OpenAI

        self.model_name = model or os.environ.get(
            "NVIDIA_VLM_MODEL", "meta/llama-3.2-90b-vision-instruct"
        )
        self._client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key or os.environ.get("NVIDIA_API_KEY", "dummy"),
        )

    def caption(self, image_bytes: bytes) -> str:
        image_b64 = base64.b64encode(image_bytes).decode()
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            max_tokens=250,
        )
        return response.choices[0].message.content


# --------------------------------------------------------------------------
# Caption cache — keyed by image content hash, so identical images
# (re-ingested docs, repeated logos/figures) never re-pay VLM cost.
# --------------------------------------------------------------------------

class CaptionCache(ABC):
    @abstractmethod
    def get(self, key: str) -> str | None:
        ...

    @abstractmethod
    def set(self, key: str, value: str) -> None:
        ...

    def flush(self) -> None:  # override if the backend needs an explicit flush
        pass


class FileCaptionCache(CaptionCache):
    """JSON-file-backed cache. Swap for Redis/SQLite at higher volume."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._dirty = False
        self._data: dict[str, str] = {}
        if self._path.exists():
            self._data = json.loads(self._path.read_text())

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value
        self._dirty = True

    def flush(self) -> None:
        if self._dirty:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._data))
            self._dirty = False


# --------------------------------------------------------------------------
# Asset storage — original images live on disk/object storage, not in the
# chunk JSON. raw_asset_b64 is only ever used transiently in memory for a
# model call and is never written to the persisted Chunk.
# --------------------------------------------------------------------------

class AssetStore(ABC):
    @abstractmethod
    async def save(self, image_bytes: bytes, source_doc_id: str, element_idx: int) -> str:
        """Persist the image and return an asset path/URI."""
        ...


class AsyncGridFSAssetStore(AssetStore):
    def __init__(self) -> None:
        pass

    async def save(self, image_bytes: bytes, source_doc_id: str, element_idx: int) -> str:
        from app.database.mongodb import fs
        filename = f"{source_doc_id}_{element_idx}.png"
        file_id = await fs.upload_from_stream(
            filename=filename,
            source=image_bytes,
            metadata={"contentType": "image/png"}
        )
        return f"gridfs://{file_id}"


# Swap in an S3/GCS-backed AssetStore for production — same interface,
# `save()` returns an s3:// or gs:// URI instead of a local path.


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def resize_image(image_bytes: bytes, max_dimension: int) -> bytes:
    """Downscale so neither side exceeds max_dimension. No-op if already small."""
    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_dimension / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def classify_visual(ocr_text: str) -> str:
    """
    Cheap heuristic to route captioning spend: chart/diagram get a VLM
    caption, plain photos don't. Swap for a real classifier (or a single
    cheap VLM call with a classification-only prompt) if precision here
    matters more than it costs.
    """
    word_count = len(ocr_text.split())
    if word_count == 0:
        return "photo"
    digit_ratio = sum(c.isdigit() for c in ocr_text) / max(len(ocr_text), 1)
    keyword_hit = any(k in ocr_text.lower() for k in ("%", "axis", "fig.", "figure", "chart"))
    if digit_ratio > 0.04 or keyword_hit:
        return "chart"
    return "diagram" if word_count > 6 else "photo"


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------

class MultimodalDocumentParser:
    def __init__(
        self,
        ocr_engine: OCREngine,
        captioner: ImageCaptioner,
        asset_store: AssetStore,
        caption_cache: CaptionCache | None = None,
        caption_images: bool = True,
        enable_ocr: bool = True,
        max_image_dimension: int = 1600,
        caption_visual_types: tuple[str, ...] = ("chart", "diagram"),
    ) -> None:
        self._ocr_engine = ocr_engine
        self._captioner = captioner
        self._asset_store = asset_store
        self._caption_cache = caption_cache
        self._caption_images = caption_images
        self._enable_ocr = enable_ocr
        self._max_image_dimension = max_image_dimension
        self._caption_visual_types = caption_visual_types
        self.issues: list[ProcessingIssue] = []

    async def parse(self, file_path: str, source_doc_id: str) -> list[Chunk]:
        elements = self._partition(file_path)

        chunks: list[Chunk] = []
        text_elements: list[Element] = []

        for idx, el in enumerate(elements):
            page = getattr(el.metadata, "page_number", None)
            try:
                if isinstance(el, Table):
                    chunk = await self._build_table_chunk(el, page, source_doc_id, idx, file_path)
                    chunks.append(chunk)
                elif isinstance(el, Image) or el.category == "Image":
                    chunk = await self._build_image_chunk(el, page, source_doc_id, idx, file_path)
                    chunks.append(chunk)
                else:
                    text_elements.append(el)
            except Exception as e:
                # One bad table/image must not take down the whole document.
                self._log_issue(source_doc_id, idx, "element_routing", "error", str(e), page)

        try:
            chunks.extend(self._build_text_chunks(text_elements, source_doc_id, len(chunks)))
        except Exception as e:
            self._log_issue(source_doc_id, -1, "text_chunking", "error", str(e))

        if self._caption_cache:
            self._caption_cache.flush()

        return chunks

    def caption_on_demand(self, chunk: Chunk) -> str:
        """
        Caption a chunk that wasn't auto-captioned at ingest (e.g. a
        "photo"-classified image) — call this when a user's question
        specifically needs a visual description of that asset. Result is
        cached the same way ingest-time captions are.
        """
        asset_path = chunk.metadata.get("asset_path")
        if not asset_path or not os.path.exists(asset_path):
            raise FileNotFoundError(f"No stored asset for chunk {chunk.chunk_id}")

        image_bytes = Path(asset_path).read_bytes()
        cache_key = hashlib.sha256(image_bytes).hexdigest()

        if self._caption_cache:
            cached = self._caption_cache.get(cache_key)
            if cached is not None:
                return cached

        caption = self._captioner.caption(image_bytes)
        if self._caption_cache:
            self._caption_cache.set(cache_key, caption)
            self._caption_cache.flush()
        return caption

    # -- partitioning ------------------------------------------------------

    @staticmethod
    def _partition(file_path: str) -> list[Element]:
        """
        hi_res strategy detects layout, table structure, and embedded
        images across PDF/DOCX/PPTX/HTML/image inputs that Unstructured
        supports. Born-digital pages pull the native text layer directly
        and never touch OCR; scanned pages fall back to Unstructured's
        own default OCR_AGENT at the page level (RapidOCR is applied
        explicitly to extracted Image/Table crops below).
        """
        return partition(
            filename=file_path,
            strategy="hi_res",
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True,
            infer_table_structure=True,
        )

    # -- table handling ------------------------------------------------------

    async def _build_table_chunk(
        self, el: Table, page: int | None, source_doc_id: str, idx: int, file_path: str
    ) -> Chunk:
        citation_id = f"{source_doc_id}:p{page}:table:{idx}"
        html = getattr(el.metadata, "text_as_html", None)
        raw_b64 = getattr(el.metadata, "image_base64", None)

        ocr_text = ""
        asset_path = None

        if raw_b64:
            image_bytes = resize_image(base64.b64decode(raw_b64), self._max_image_dimension)

            # Only OCR the table crop if Unstructured couldn't extract a
            # structured HTML representation — covers image-only tables.
            if not html and self._enable_ocr:
                try:
                    ocr_text = self._ocr_engine.extract_text(image_bytes)
                except Exception as e:
                    self._log_issue(source_doc_id, idx, "ocr", "warning", str(e), page)

            try:
                asset_path = await self._asset_store.save(image_bytes, source_doc_id, idx)
            except Exception as e:
                self._log_issue(source_doc_id, idx, "asset_store", "warning", str(e), page)

        text_representation = html or ocr_text or (el.text or "")

        return Chunk(
            chunk_id=citation_id,
            source_doc_id=source_doc_id,
            page_number=page,
            modality="table",
            text_representation=text_representation,
            metadata={
                "element_type": "Table",
                "source_filename": file_path,
                "citation_id": citation_id,
                "ocr_text": ocr_text,
                "used_ocr_fallback": bool(ocr_text and not html),
                "asset_path": asset_path,
            },
        )

    # -- image handling ------------------------------------------------------

    async def _build_image_chunk(
        self, el: Element, page: int | None, source_doc_id: str, idx: int, file_path: str
    ) -> Chunk:
        citation_id = f"{source_doc_id}:p{page}:image:{idx}"
        raw_b64 = getattr(el.metadata, "image_base64", None)
        if not raw_b64:
            raise ValueError("Image element has no embedded payload to process")

        image_bytes = resize_image(base64.b64decode(raw_b64), self._max_image_dimension)

        ocr_text = ""
        if self._enable_ocr:
            try:
                ocr_text = self._ocr_engine.extract_text(image_bytes)
            except Exception as e:
                self._log_issue(source_doc_id, idx, "ocr", "warning", str(e), page)

        visual_type = classify_visual(ocr_text)

        caption = ""
        caption_generated = False
        should_caption = self._caption_images and visual_type in self._caption_visual_types
        if should_caption:
            caption, caption_generated = self._get_or_generate_caption(
                image_bytes, source_doc_id, idx, page
            )

        asset_path = None
        try:
            asset_path = await self._asset_store.save(image_bytes, source_doc_id, idx)
        except Exception as e:
            self._log_issue(source_doc_id, idx, "asset_store", "warning", str(e), page)

        text_representation = caption
        if ocr_text.strip():
            text_representation = f"{text_representation}\n\nText in image:\n{ocr_text}".strip()
        if not text_representation:
            text_representation = f"[{visual_type} image — no extractable text or caption]"

        return Chunk(
            chunk_id=citation_id,
            source_doc_id=source_doc_id,
            page_number=page,
            modality="image",
            text_representation=text_representation,
            metadata={
                "element_type": "Image",
                "visual_type": visual_type,
                "source_filename": file_path,
                "citation_id": citation_id,
                "ocr_text": ocr_text,
                "caption_generated": caption_generated,
                "caption_model": self._captioner.model_name if caption_generated else None,
                "asset_path": asset_path,
            },
        )

    def _get_or_generate_caption(
        self, image_bytes: bytes, source_doc_id: str, idx: int, page: int | None
    ) -> tuple[str, bool]:
        cache_key = hashlib.sha256(image_bytes).hexdigest()

        if self._caption_cache:
            cached = self._caption_cache.get(cache_key)
            if cached is not None:
                return cached, True

        try:
            caption = self._captioner.caption(image_bytes)
        except Exception as e:
            self._log_issue(source_doc_id, idx, "vlm_caption", "warning", str(e), page)
            return "", False

        if self._caption_cache:
            self._caption_cache.set(cache_key, caption)

        return caption, True

    # -- text handling ------------------------------------------------------

    @staticmethod
    def _build_text_chunks(
        text_elements: list[Element], source_doc_id: str, start_idx: int
    ) -> list[Chunk]:
        text_chunks = chunk_by_title(
            text_elements,
            max_characters=1500,
            combine_text_under_n_chars=300,
            new_after_n_chars=1200,
        )

        chunks = []
        for i, tc in enumerate(text_chunks):
            page = getattr(tc.metadata, "page_number", None)
            citation_id = f"{source_doc_id}:p{page}:text:{start_idx + i}"
            chunks.append(
                Chunk(
                    chunk_id=citation_id,
                    source_doc_id=source_doc_id,
                    page_number=page,
                    modality="text",
                    text_representation=tc.text,
                    metadata={"element_type": "Text", "citation_id": citation_id},
                )
            )
        return chunks

    # -- error handling ------------------------------------------------------

    def _log_issue(
        self,
        source_doc_id: str,
        element_index: int,
        stage: str,
        severity: str,
        message: str,
        page: int | None = None,
    ) -> None:
        issue = ProcessingIssue(source_doc_id, element_index, stage, severity, message, page)
        self.issues.append(issue)
        log_fn = logger.error if severity == "error" else logger.warning
        log_fn("[%s] doc=%s element=%s page=%s: %s", stage, source_doc_id, element_index, page, message)


# --------------------------------------------------------------------------
# Output writing — JSONL for chunks so large documents can be indexed
# incrementally without loading the whole file into memory.
# --------------------------------------------------------------------------

def write_outputs(chunks: list[Chunk], issues: list[ProcessingIssue], output_prefix: str) -> None:
    chunks_path = f"{output_prefix}.chunks.jsonl"
    issues_path = f"{output_prefix}.issues.jsonl"
    manifest_path = f"{output_prefix}.manifest.json"

    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")

    with open(issues_path, "w", encoding="utf-8") as f:
        for i in issues:
            f.write(json.dumps(asdict(i)) + "\n")

    manifest = {
        "generated_at": time.time(),
        "chunk_count": len(chunks),
        "by_modality": {m: sum(c.modality == m for c in chunks) for m in ("text", "table", "image")},
        "captions_generated": sum(c.metadata.get("caption_generated") is True for c in chunks),
        "issue_count": len(issues),
        "error_count": sum(i.severity == "error" for i in issues),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(chunks)} chunks    -> {chunks_path}")
    print(f"Wrote {len(issues)} issues    -> {issues_path}")
    print(f"Wrote manifest              -> {manifest_path}")


async def run(
    file_path: str,
    output_dir: str,
    notebook_id: str = "",
    dry_run: bool = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    source_doc_id = os.path.splitext(os.path.basename(file_path))[0]

    parser = MultimodalDocumentParser(
        ocr_engine=RapidOCREngine(),
        captioner=NvidiaVLMCaptioner(),
        asset_store=AsyncGridFSAssetStore(),
        caption_cache=FileCaptionCache(os.path.join(output_dir, "caption_cache.json")),
        caption_images=True,          # master on/off switch for all VLM calls
        enable_ocr=True,
        max_image_dimension=1600,     # caps both OCR/VLM cost and stored asset size
        caption_visual_types=("chart", "diagram"),  # plain photos never get captioned
    )

    chunks = await parser.parse(file_path, source_doc_id)
    write_outputs(chunks, parser.issues, os.path.join(output_dir, source_doc_id))

    # ── Azure embedding & indexing ─────────────────────────────────────
    # Skip gracefully if Azure env vars are not set — the existing
    # pipeline still produces JSONL output without Azure configured.
    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        logger.warning(
            "Azure not configured — skipping embedding & indexing. "
            "Set AZURE_OPENAI_ENDPOINT and related env vars to enable."
        )
    else:
        logger.info("Starting Azure embedding & indexing...")
        result = await index_chunks_to_cosmos(chunks, notebook_id=notebook_id, dry_run=dry_run)
        if dry_run:
            logger.info("Dry run result: %s", result)
        else:
            logger.info(
                "Azure indexing complete: %d/%d documents uploaded "
                "(text_vectors=%d, image_vectors=%d)",
                result["uploaded"], result["total_documents"],
                result["text_vectors"], result["image_vectors"],
            )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python prasing.py <input_file> <output_dir> [--dry-run]")
        sys.exit(1)

    dry_run = "--dry-run" in sys.argv
    asyncio.run(run(sys.argv[1], sys.argv[2], dry_run=dry_run))
