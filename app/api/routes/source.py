from fastapi import APIRouter, HTTPException, status
from app.api.schemas.source import SourceCreate, SourceResponse
from app.database.mongodb import sources_collection
from datetime import datetime, timezone
import logging

from bson import ObjectId
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Depends

from app.database.mongodb import (
    notebooks_collection,
    sources_collection,
    fs,
)
from app.api.routes.components.security import get_current_user


router = APIRouter(
    prefix="/notebooks/{notebook_id}/sources",
    tags=["Sources"],
    dependencies=[Depends(get_current_user)]
)


@router.post(
    "",
    response_model=SourceResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_source(
    notebook_id: str,
    data: SourceCreate,
):
    # 1. Validate notebook ID
    if not ObjectId.is_valid(notebook_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid notebook ID",
        )

    # 2. Check notebook exists
    notebook = await notebooks_collection.find_one(
        {"_id": ObjectId(notebook_id)}
    )

    if notebook is None:
        raise HTTPException(
            status_code=404,
            detail="Notebook not found",
        )

    # 3. Create source
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

    # 4. Save source
    result = await sources_collection.insert_one(source)

    # 5. Return API response
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


@router.post(
    "/upload",
    response_model=SourceResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_source(
    notebook_id: str,
    file: UploadFile = File(...),
):
    # 1. Validate notebook ID
    if not ObjectId.is_valid(notebook_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid notebook ID",
        )

    # 2. Check notebook exists
    notebook = await notebooks_collection.find_one(
        {"_id": ObjectId(notebook_id)}
    )

    if notebook is None:
        raise HTTPException(
            status_code=404,
            detail="Notebook not found",
        )

    # 3. Upload file to GridFS
    try:
        async with fs.open_upload_stream(file.filename) as upload_stream:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                await upload_stream.write(chunk)
            file_id = upload_stream._id
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload file: {str(e)}"
        )

    # 4. Create source
    now = datetime.now(timezone.utc)
    
    # Extract file extension for source_type if applicable
    source_type = "file"
    if file.filename:
        ext = file.filename.split(".")[-1].lower()
        if ext in ["pdf", "pptx", "doc", "docx", "txt"]:
            source_type = ext

    source = {
        "notebook_id": ObjectId(notebook_id),
        "name": file.filename,
        "file_id": file_id,
        "source_type": source_type,
        "status": "pending",
        "created_at": now,
        "updated_at": now,
    }

    # 5. Save source
    result = await sources_collection.insert_one(source)

    # 6. Auto-parse: download from GridFS and run the parsing pipeline
    import asyncio
    import os
    import tempfile

    logger = logging.getLogger(__name__)

    async def _parse_in_background():
        """Background task: download file from GridFS → parse → index."""
        try:
            # Update status
            await sources_collection.update_one(
                {"_id": result.inserted_id},
                {"$set": {"status": "processing"}}
            )

            # Download from GridFS to a temp file
            from app.database.mongodb import fs
            import io

            tmp_dir = tempfile.mkdtemp()
            tmp_path = os.path.join(tmp_dir, file.filename or "document")

            buf = io.BytesIO()
            await fs.download_to_stream(file_id, buf)
            with open(tmp_path, "wb") as f:
                f.write(buf.getvalue())

            # Run the parsing + indexing pipeline
            from app.api.routes.components.prasing.prasing import run as parse_run
            await parse_run(
                file_path=tmp_path,
                output_dir=tmp_dir,
                notebook_id=notebook_id,
            )

            # Update status to ready
            await sources_collection.update_one(
                {"_id": result.inserted_id},
                {"$set": {"status": "ready"}}
            )
            logger.info("Auto-parse complete for source: %s", result.inserted_id)

        except Exception as e:
            logger.error("Auto-parse failed for source %s: %s", result.inserted_id, e)
            await sources_collection.update_one(
                {"_id": result.inserted_id},
                {"$set": {"status": "failed", "error": str(e)}}
            )

    asyncio.create_task(_parse_in_background())

    # 7. Return API response
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