from sqlalchemy import Column, String, Integer, Float, JSON, ForeignKey, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import enum

from .base import BaseModel


class DatasetStatus(str, enum.Enum):
    """
    Dataset processing status
    """
    CREATED = "created"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Dataset(BaseModel):
    """
    Dataset model for storing dataset metadata
    """
    __tablename__ = "datasets"
    
    name = Column(String, nullable=False)
    description = Column(String)
    status = Column(Enum(DatasetStatus), default=DatasetStatus.CREATED)
    
    # Data source information
    source_type = Column(String)  # csv, json, database, api, etc.
    source_config = Column(JSON)  # Connection details, file paths, etc.
    
    # Dataset statistics
    row_count = Column(Integer, default=0)
    column_count = Column(Integer, default=0)
    size_bytes = Column(Integer, default=0)
    
    # Schema and metadata
    schema = Column(JSON)  # Column names, types, etc.
    statistics = Column(JSON)  # Min, max, mean, null counts, etc.
    quality_score = Column(Float)  # 0-1 quality assessment score
    
    # Storage location
    storage_path = Column(String)  # S3 path or local path
    
    # Owner
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    owner = relationship("User", back_populates="datasets")
    
    # Related pipelines
    pipeline_executions = relationship("PipelineExecution", back_populates="dataset")