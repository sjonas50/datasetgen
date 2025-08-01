"""
Database configuration and models
"""

import os
from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Text, Float, Integer, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# Database URL
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://platform_user:platform_pass@localhost:5432/dataset_platform")
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Create base class
Base = declarative_base()

# Create engines
engine = create_engine(DATABASE_URL, echo=True)
async_engine = create_async_engine(ASYNC_DATABASE_URL, echo=True)

# Create session factories
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
AsyncSessionLocal = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

# Database models
class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    files = relationship("File", back_populates="owner", cascade="all, delete-orphan")
    datasets = relationship("Dataset", back_populates="owner", cascade="all, delete-orphan")
    pipelines = relationship("Pipeline", back_populates="owner", cascade="all, delete-orphan")

class File(Base):
    __tablename__ = "files"
    
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    size = Column(Integer, nullable=False)
    preview_data = Column(JSON)
    owner_id = Column(String, ForeignKey("users.id"), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    owner = relationship("User", back_populates="files")
    dataset_files = relationship("DatasetFile", back_populates="file", cascade="all, delete-orphan")

class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    schema_json = Column(JSON)
    row_count = Column(Integer)
    column_count = Column(Integer)
    quality_score = Column(Float)
    quality_report = Column(JSON)
    owner_id = Column(String, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    owner = relationship("User", back_populates="datasets")
    dataset_files = relationship("DatasetFile", back_populates="dataset", cascade="all, delete-orphan")
    pipelines = relationship("Pipeline", back_populates="dataset")

class DatasetFile(Base):
    __tablename__ = "dataset_files"
    
    dataset_id = Column(String, ForeignKey("datasets.id"), primary_key=True)
    file_id = Column(String, ForeignKey("files.id"), primary_key=True)
    added_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="dataset_files")
    file = relationship("File", back_populates="dataset_files")

class Pipeline(Base):
    __tablename__ = "pipelines"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=False)
    steps = Column(JSON, nullable=False)  # Array of step configurations
    status = Column(String, default="draft")  # draft, active, paused, archived
    owner_id = Column(String, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_execution_at = Column(DateTime)
    
    # Relationships
    owner = relationship("User", back_populates="pipelines")
    dataset = relationship("Dataset", back_populates="pipelines")
    executions = relationship("PipelineExecution", back_populates="pipeline", cascade="all, delete-orphan")

class PipelineExecution(Base):
    __tablename__ = "pipeline_executions"
    
    id = Column(String, primary_key=True)
    pipeline_id = Column(String, ForeignKey("pipelines.id"), nullable=False)
    status = Column(String, nullable=False)  # pending, running, completed, failed
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    error = Column(Text)
    results = Column(JSON)  # Step results
    metrics = Column(JSON)  # Execution metrics (duration, records processed, etc.)
    output_dataset_id = Column(String, ForeignKey("datasets.id"))
    
    # Relationships
    pipeline = relationship("Pipeline", back_populates="executions")
    output_dataset = relationship("Dataset", foreign_keys=[output_dataset_id])

class ProcessingLog(Base):
    __tablename__ = "processing_logs"
    
    id = Column(String, primary_key=True)
    execution_id = Column(String, ForeignKey("pipeline_executions.id"), nullable=False)
    step_name = Column(String, nullable=False)
    status = Column(String, nullable=False)
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    input_records = Column(Integer)
    output_records = Column(Integer)
    errors = Column(JSON)
    warnings = Column(JSON)
    metrics = Column(JSON)
    
    # Relationships
    execution = relationship("PipelineExecution")

# Helper functions
def get_db() -> Session:
    """Get synchronous database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_async_db() -> AsyncSession:
    """Get asynchronous database session"""
    async with AsyncSessionLocal() as session:
        yield session

def create_tables():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)

def drop_tables():
    """Drop all tables"""
    Base.metadata.drop_all(bind=engine)