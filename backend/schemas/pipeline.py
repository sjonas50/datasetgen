from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any, List
import uuid

from models.pipeline import PipelineStatus


class PipelineStep(BaseModel):
    """
    Individual pipeline step configuration
    """
    name: str
    type: str  # transform, filter, aggregate, llm, etc.
    config: Dict[str, Any]
    depends_on: Optional[List[str]] = []  # Step dependencies
    position: Optional[Dict[str, float]] = None  # Visual position


class PipelineBase(BaseModel):
    """
    Base pipeline schema
    """
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    config: Dict[str, Any]  # Full pipeline configuration
    llm_config: Optional[Dict[str, Any]] = None


class PipelineCreate(PipelineBase):
    """
    Schema for creating a pipeline
    """
    steps: List[PipelineStep]


class PipelineUpdate(BaseModel):
    """
    Schema for updating a pipeline
    """
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[PipelineStatus] = None
    config: Optional[Dict[str, Any]] = None
    llm_config: Optional[Dict[str, Any]] = None


class PipelineResponse(PipelineBase):
    """
    Schema for pipeline response
    """
    id: uuid.UUID
    status: PipelineStatus
    is_scheduled: bool
    schedule_config: Optional[Dict[str, Any]]
    owner_id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class PipelineExecute(BaseModel):
    """
    Schema for pipeline execution request
    """
    dataset_id: Optional[uuid.UUID] = None
    initial_data: Optional[Any] = None  # For direct data input
    parameters: Optional[Dict[str, Any]] = {}
    dry_run: bool = False