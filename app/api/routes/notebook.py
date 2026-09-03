from datetime import datetime, timezone
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from bson import ObjectId
from bson.errors import InvalidId

from app.database.mongodb import notebooks_collection
from app.api.schemas.notebook import NotebookCreate, NotebookRename
from app.models.notebook import Notebook as NotebookResponse
from app.api.routes.components.security import get_current_user

router = APIRouter(dependencies=[Depends(get_current_user)])


@router.post("",response_model=NotebookResponse,status_code=201, response_model_by_alias=False)
async def create_notebook(data: NotebookCreate):
    now = datetime.now(timezone.utc)

    notebook = {
        "title": data.title,
        "description": data.description,
        "created_at": now,
        "updated_at": now,
        "tags": data.tags,
    }

    result = await notebooks_collection.insert_one(notebook)
    
    return NotebookResponse(
        id=str(result.inserted_id),
        title=data.title,
        description=data.description,
        created_at=now,
        updated_at=now,
        tags=data.tags,
    )


@router.delete("/{notebook_id}")
async def delete_notebook(notebook_id: str):
    try:
        obj_id = ObjectId(notebook_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid Notebook ID format")

    result = await notebooks_collection.delete_one({"_id": obj_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Notebook not found")
    
    return {"message": "Notebook deleted successfully"}

@router.get("", response_model=List[NotebookResponse], response_model_by_alias=False)
async def get_all_notebooks():
    notebooks = await notebooks_collection.find().to_list(100)
    
    for notebook in notebooks:
        notebook["id"] = str(notebook.pop("_id"))
        
    return notebooks


@router.patch("/{notebook_id}", response_model=NotebookResponse, response_model_by_alias=False)
async def rename_notebook(notebook_id: str, data: NotebookRename):
    try:
        obj_id = ObjectId(notebook_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid Notebook ID format")

    now = datetime.now(timezone.utc)
    result = await notebooks_collection.update_one(
        {"_id": obj_id},
        {"$set": {"title": data.title, "updated_at": now}}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Notebook not found")

    updated_notebook = await notebooks_collection.find_one({"_id": obj_id})
    updated_notebook["id"] = str(updated_notebook.pop("_id"))
    
    return updated_notebook
        