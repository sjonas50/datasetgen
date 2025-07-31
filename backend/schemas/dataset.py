from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any, List
import uuid

from models.dataset import DatasetStatus


class DatasetBase(BaseModel):
    """
    Base dataset schema
    """
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    source_type: str
    source_config: Dict[str, Any]


class DatasetCreate(DatasetBase):
    """
    Schema for creating a dataset
    """
    pass


class DatasetUpdate(BaseModel):
    """
    Schema for updating a dataset
    """
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[DatasetStatus] = None


class DatasetResponse(DatasetBase):
    """
    Schema for dataset response
    """
    id: uuid.UUID
    status: DatasetStatus
    row_count: int
    column_count: int
    size_bytes: int
    schema: Optional[Dict[str, Any]]
    statistics: Optional[Dict[str, Any]]
    quality_score: Optional[float]
    storage_path: Optional[str]
    owner_id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class DatasetList(BaseModel):
    """
    Schema for paginated dataset list
    """
    datasets: List[DatasetResponse]
    total: int
    skip: int
    limit: int