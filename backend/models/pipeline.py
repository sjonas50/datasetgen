from sqlalchemy import Column, String, JSON, ForeignKey, Enum, Boolean, Integer, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import enum

from .base import BaseModel


class PipelineStatus(str, enum.Enum):
    """
    Pipeline status
    """
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class Pipeline(BaseModel):
    """
    Data processing pipeline model
    """
    __tablename__ = "pipelines"
    
    name = Column(String, nullable=False)
    description = Column(String)
    status = Column(Enum(PipelineStatus), default=PipelineStatus.DRAFT)
    
    # Pipeline configuration
    config = Column(JSON, nullable=False)  # Steps, transformations, etc.
    
    # LLM configuration
    llm_config = Column(JSON)  # Model selection, prompts, etc.
    
    # Scheduling
    is_scheduled = Column(Boolean, default=False)
    schedule_config = Column(JSON)  # Cron expression, frequency, etc.
    
    # Owner
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    owner = relationship("User", back_populates="pipelines")
    
    # Executions
    executions = relationship("PipelineExecution", back_populates="pipeline", cascade="all, delete-orphan")


class PipelineExecution(BaseModel):
    """
    Pipeline execution history
    """
    __tablename__ = "pipeline_executions"
    
    # Pipeline reference
    pipeline_id = Column(UUID(as_uuid=True), ForeignKey("pipelines.id"), nullable=False)
    pipeline = relationship("Pipeline", back_populates="executions")
    
    # Dataset reference
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"))
    dataset = relationship("Dataset", back_populates="pipeline_executions")
    
    # Execution details
    status = Column(String, nullable=False)  # running, completed, failed
    started_at = Column(String)
    completed_at = Column(String)
    error_message = Column(String)
    
    # Execution metrics
    rows_processed = Column(Integer, default=0)
    execution_time_seconds = Column(Float)
    cost_dollars = Column(Float)
    
    # Results
    output_dataset_id = Column(UUID(as_uuid=True))
    logs = Column(JSON)