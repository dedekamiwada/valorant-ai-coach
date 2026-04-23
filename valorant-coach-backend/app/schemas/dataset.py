from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    source: str = "user"  # "user" or "pro"
    player_name: Optional[str] = None
    team: Optional[str] = None
    agent: Optional[str] = None
    map_name: Optional[str] = None
    rank: Optional[str] = None
    tags: Optional[list[str]] = None


class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    source: Optional[str] = None
    player_name: Optional[str] = None
    team: Optional[str] = None
    agent: Optional[str] = None
    map_name: Optional[str] = None
    rank: Optional[str] = None
    tags: Optional[list[str]] = None


class DatasetResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    source: str
    player_name: Optional[str] = None
    team: Optional[str] = None
    agent: Optional[str] = None
    map_name: Optional[str] = None
    rank: Optional[str] = None
    tags: Optional[list[str]] = None
    filename: str
    file_size_bytes: Optional[int] = None
    duration_seconds: Optional[float] = None
    analysis_id: Optional[str] = None
    status: str
    analysis_progress: Optional[int] = None
    analysis_status_text: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DatasetListItem(BaseModel):
    id: str
    name: str
    source: str
    player_name: Optional[str] = None
    agent: Optional[str] = None
    map_name: Optional[str] = None
    filename: str
    file_size_bytes: Optional[int] = None
    duration_seconds: Optional[float] = None
    analysis_id: Optional[str] = None
    status: str
    analysis_progress: Optional[int] = None
    analysis_status_text: Optional[str] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DatasetUploadResponse(BaseModel):
    id: str
    message: str
