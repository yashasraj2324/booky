from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from datetime import datetime

class Source(BaseModel):
    id: str = Field(alias="_id", default=None)
    notebook_id: str
    name: str
    url: Optional[str] = None
    source_type: str              # youtube, pdf, web, text
    status: str = "pending"       # pending, processing, ready, failed
    transcript: Optional[str] = None
    
    model_config = ConfigDict(populate_by_name=True)


class Notebook(BaseModel):
    id: str = Field(alias="_id", default=None)
    user_id: str = Field(default="", description="Owner's user ID")
    title: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    sources: List[Source] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(populate_by_name=True)