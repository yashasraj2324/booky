from pydantic import BaseModel, Field
from typing import List, Optional

class NotebookCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str | None = None
    tags: list[str] = Field(default_factory=list)


class NotebookRename(BaseModel):
    title: str = Field(min_length=1, max_length=200)