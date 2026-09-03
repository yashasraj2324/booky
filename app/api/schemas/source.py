from datetime import datetime

from pydantic import BaseModel, Field


class SourceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    url: str | None = None
    text: str | None = None
    source_type: str


class SourceResponse(BaseModel):
    id: str
    notebook_id: str
    file_id: str | None = None
    name: str
    url: str | None = None
    text: str | None = None
    source_type: str
    status: str
    created_at: datetime
    updated_at: datetime