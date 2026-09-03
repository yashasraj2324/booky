"""Sources API — upload files and add sources to notebooks.

Each endpoint verifies that the notebook belongs to the current user before
allowing access.  File uploads trigger a background parsing + indexing task.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status

from app.api.schemas.source import SourceCreate, SourceResponse
from app.api.routes.components.security import get_current_user
from app.database.mongodb import fs, notebooks_collection, sources_collection
from app.models.users import UserResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/notebooks/{notebook_id}/sources",
    tags=["Sources"],
    dependencies=[Depends(get_current_user)],
)


async def _verify_notebook_ownership(
    notebook_id: str, current_user: UserResponse
) -> ObjectId:
    """Validate notebook ID and verify the current user owns it."""
    if not ObjectId.is_valid(notebook_id):
        raise HTTPException(status_code=400, detail="Invalid notebook ID")

    obj_id = ObjectId(notebook_id)
    notebook = await notebooks_collection.find_one(
        {"_id": obj_id, "user_id": str(current_user.id)}
    )
    if notebook is None:
        raise HTTPException(
            status_code=404,
            detail="Notebook not found or not owned by you",
        )
    return obj_id


@router.post("", response_model=SourceResponse, status_code=status.HTTP_201_CREATED)
async def add_source(
    notebook_id: str,
    data: SourceCreate,
    current_user: UserResponse = Depends(get_current_user),
):
    await _verify_notebook_ownership(notebook_id, current_user)

    now = datetime.now(timezone.utc)
    source = {
        "notebook_id": ObjectId(notebook_id),
        "name": data.name,
        "url": data.url,
        "text": data.text,
        "source_type": data.source_type,
        "status": "pending",
        "created_at": now,
        "updated_at": now,
    }

    result = await sources_collection.insert_one(source)

    return SourceResponse(
        id=str(result.inserted_id),
        notebook_id=notebook_id,
        name=data.name,
        url=data.url,
        text=data.text,
        source_type=data.source_type,
        status="pending",
        created_at=now,
        updated_at=now,
    )


@router.post("/upload", response_model=SourceResponse, status_code=status.HTTP_201_CREATED)
async def upload_source(
    notebook_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: UserResponse = Depends(get_current_user),
):
    await _verify_notebook_ownership(notebook_id, current_user)

    # Upload file to GridFS
    try:
        async with fs.open_upload_stream(file.filename) as upload_stream:
            while chunk := await file.read(1024 * 1024):
                await upload_stream.write(chunk)
            file_id = upload_stream._id
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload file: {str(e)}",
        )

    # Determine source_type from file extension
    source_type = "file"
    if file.filename:
        ext = file.filename.split(".")[-1].lower()
        if ext in ["pdf", "pptx", "doc", "docx", "txt"]:
            source_type = ext

    now = datetime.now(timezone.utc)
    source = {
        "notebook_id": ObjectId(notebook_id),
        "name": file.filename,
        "file_id": file_id,
        "source_type": source_type,
        "status": "pending",
        "created_at": now,
        "updated_at": now,
    }

    result = await sources_collection.insert_one(source)

    # Schedule background parsing + indexing
    background_tasks.add_task(
        _parse_in_background,
        source_id=result.inserted_id,
        file_id=file_id,
        filename=file.filename or "document",
        notebook_id=notebook_id,
    )

    return SourceResponse(
        id=str(result.inserted_id),
        notebook_id=notebook_id,
        file_id=str(file_id),
        name=file.filename or "uploaded_file",
        source_type=source_type,
        status="pending",
        created_at=now,
        updated_at=now,
    )


async def _parse_in_background(
    source_id: ObjectId,
    file_id: ObjectId,
    filename: str,
    notebook_id: str,
) -> None:
    """Background task: download from GridFS → parse → index → cleanup."""
    # Lazy import to avoid circular dependency at module load time
    from app.api.routes.components.prasing.prasing import run as parse_run

    tmp_dir = tempfile.mkdtemp()
    try:
        # Update source status
        await sources_collection.update_one(
            {"_id": source_id},
            {"$set": {"status": "processing"}},
        )

        # Download from GridFS to a temp file
        tmp_path = os.path.join(tmp_dir, filename)
        buf = io.BytesIO()
        await fs.download_to_stream(file_id, buf)
        with open(tmp_path, "wb") as f:
            f.write(buf.getvalue())

        # Run the parsing + indexing pipeline
        await parse_run(
            file_path=tmp_path,
            output_dir=tmp_dir,
            notebook_id=notebook_id,
        )

        # Update source status to ready
        await sources_collection.update_one(
            {"_id": source_id},
            {"$set": {"status": "ready"}},
        )
        logger.info("Auto-parse complete for source: %s", source_id)

    except Exception as e:
        logger.error("Auto-parse failed for source %s: %s", source_id, e)
        await sources_collection.update_one(
            {"_id": source_id},
            {"$set": {"status": "failed", "error": str(e)}},
        )

    finally:
        # Always clean up temp files, even on failure
        shutil.rmtree(tmp_dir, ignore_errors=True)
