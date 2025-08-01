"""
Advanced DatasetGen Backend with Database and AI Integration
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, status, Form, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import selectinload
import pandas as pd
import aiofiles
import mimetypes
import jwt
import bcrypt
from dotenv import load_dotenv
from datetime import timedelta
import json

# Load environment variables
load_dotenv()

# Import database models and utilities
from database import (
    get_async_db, AsyncSession, create_tables,
    User, File as FileModel, Dataset, DatasetFile, Pipeline, PipelineExecution
)
from services.claude_service import ClaudeService
from celery_app import celery_app
from tasks.pipeline_tasks import execute_pipeline, analyze_dataset_with_ai
from pydantic import BaseModel, Field

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Pydantic schemas
class UserCreate(BaseModel):
    email: str
    username: str
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    is_active: bool
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    file_type: str
    size: int
    uploaded_at: datetime
    preview: Optional[Dict[str, Any]] = None

class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    file_ids: List[str]

class DatasetResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    schema_json: Optional[Dict[str, Any]] = Field(None, alias="schema")
    files: List[FileUploadResponse]
    owner_id: str
    created_at: datetime
    updated_at: datetime
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    quality_score: Optional[float] = None
    quality_report: Optional[Dict[str, Any]] = None

    class Config:
        populate_by_name = True

class PipelineStep(BaseModel):
    type: str
    config: Dict[str, Any]

class PipelineCreate(BaseModel):
    name: str
    description: Optional[str] = None
    dataset_id: str
    steps: List[PipelineStep]

class PipelineResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    dataset_id: str
    steps: List[PipelineStep]
    owner_id: str
    created_at: datetime
    updated_at: datetime
    last_execution_at: Optional[datetime] = None
    status: str = "draft"

class PipelineExecutionResponse(BaseModel):
    id: str
    pipeline_id: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None

class QualityReport(BaseModel):
    quality_score: float
    issues: List[Dict[str, Any]]
    patterns: List[str]
    recommendations: List[str]
    pii_concerns: List[str]
    schema_suggestions: Dict[str, Any]

# Create FastAPI app
app = FastAPI(
    title="DatasetGen API",
    description="AI-First Dataset Creation Platform with Claude Integration",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:80"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize services
claude_service = ClaudeService() if os.getenv("ANTHROPIC_API_KEY") else None

# Auth functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())

def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_async_db)
) -> User:
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    
    return user

# Utility functions
async def process_uploaded_file(file_path: Path, file_type: str) -> Dict[str, Any]:
    """Process uploaded file and extract preview data"""
    preview = {}
    
    try:
        if file_type in ["text/csv", "application/csv"]:
            df = pd.read_csv(file_path)
            preview = {
                "rows": len(df),
                "columns": list(df.columns),
                "sample_data": df.head(5).to_dict(orient="records"),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
        elif file_type == "application/json":
            with open(file_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                preview = {
                    "rows": len(data),
                    "sample_data": data[:5],
                    "structure": "array_of_objects" if isinstance(data[0], dict) else "array"
                }
            elif isinstance(data, dict):
                preview = {
                    "structure": "object",
                    "keys": list(data.keys())[:20],
                    "sample_data": {k: data[k] for k in list(data.keys())[:5]}
                }
        elif file_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
            df = pd.read_excel(file_path)
            preview = {
                "rows": len(df),
                "columns": list(df.columns),
                "sample_data": df.head(5).to_dict(orient="records"),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
        elif file_type == "application/pdf":
            # For PDFs, we'll use Claude's vision capabilities later
            preview = {
                "type": "pdf",
                "message": "PDF processing available with pipeline execution"
            }
        elif file_type in ["image/png", "image/jpeg", "image/jpg"]:
            # For images, we'll use Claude's vision capabilities
            preview = {
                "type": "image",
                "message": "Image processing available with pipeline execution"
            }
    except Exception as e:
        preview = {"error": str(e)}
    
    return preview

# Routes
@app.on_event("startup")
async def startup_event():
    """Initialize database tables"""
    # In production, use Alembic migrations instead
    create_tables()
    print("Database tables created")

@app.get("/")
async def root():
    return {
        "message": "Welcome to DatasetGen API v2.0",
        "features": {
            "ai_powered": bool(claude_service),
            "database": "PostgreSQL",
            "task_queue": "Celery",
            "file_types": ["CSV", "JSON", "Excel", "PDF", "Images"]
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "DatasetGen Backend",
        "timestamp": datetime.utcnow().isoformat(),
        "claude_available": bool(claude_service),
        "version": "2.0.0"
    }

# Auth endpoints
@app.post("/api/v1/auth/register", response_model=UserResponse)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_async_db)
):
    # Check if user exists
    result = await db.execute(
        select(User).where((User.email == user_data.email) | (User.username == user_data.username))
    )
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="User already exists")
    
    # Create user
    user = User(
        id=str(uuid.uuid4()),
        email=user_data.email,
        username=user_data.username,
        hashed_password=get_password_hash(user_data.password)
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    return UserResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        is_active=user.is_active,
        created_at=user.created_at
    )

@app.post("/api/v1/auth/login", response_model=Token)
async def login(
    username: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_async_db)
):
    # Find user
    result = await db.execute(
        select(User).where((User.email == username) | (User.username == username))
    )
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create token
    access_token = create_access_token(data={"sub": user.id})
    return Token(access_token=access_token)

# File upload endpoints
@app.post("/api/v1/files/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Upload a file for processing"""
    # Validate file type
    allowed_types = [
        "text/csv", "application/csv",
        "application/json",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "application/pdf",
        "image/png", "image/jpeg", "image/jpg"
    ]
    
    file_type = file.content_type or mimetypes.guess_type(file.filename)[0]
    if file_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_type} not supported. Allowed types: CSV, JSON, Excel, PDF, PNG, JPG"
        )
    
    # Create user directory
    user_dir = UPLOAD_DIR / current_user.id
    user_dir.mkdir(exist_ok=True)
    
    # Save file
    file_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix
    file_path = user_dir / f"{file_id}{file_extension}"
    
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    # Process file to get preview
    preview = await process_uploaded_file(file_path, file_type)
    
    # Store file metadata in database
    file_model = FileModel(
        id=file_id,
        filename=file.filename,
        file_type=file_type,
        file_path=str(file_path),
        size=len(content),
        preview_data=preview,
        owner_id=current_user.id
    )
    db.add(file_model)
    await db.commit()
    await db.refresh(file_model)
    
    return FileUploadResponse(
        file_id=file_model.id,
        filename=file_model.filename,
        file_type=file_model.file_type,
        size=file_model.size,
        uploaded_at=file_model.uploaded_at,
        preview=preview
    )

@app.get("/api/v1/files", response_model=List[FileUploadResponse])
async def list_files(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """List all uploaded files for the current user"""
    result = await db.execute(
        select(FileModel).where(FileModel.owner_id == current_user.id)
    )
    files = result.scalars().all()
    
    return [
        FileUploadResponse(
            file_id=f.id,
            filename=f.filename,
            file_type=f.file_type,
            size=f.size,
            uploaded_at=f.uploaded_at,
            preview=f.preview_data
        )
        for f in files
    ]

# Dataset endpoints
@app.get("/api/v1/datasets", response_model=List[DatasetResponse])
async def list_datasets(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    result = await db.execute(
        select(Dataset)
        .where(Dataset.owner_id == current_user.id)
        .options(selectinload(Dataset.dataset_files).selectinload(DatasetFile.file))
        .offset(skip)
        .limit(limit)
    )
    datasets = result.scalars().all()
    
    return [
        DatasetResponse(
            id=d.id,
            name=d.name,
            description=d.description,
            schema=d.schema_json,
            files=[
                FileUploadResponse(
                    file_id=df.file.id,
                    filename=df.file.filename,
                    file_type=df.file.file_type,
                    size=df.file.size,
                    uploaded_at=df.file.uploaded_at,
                    preview=df.file.preview_data
                )
                for df in d.dataset_files
            ],
            owner_id=d.owner_id,
            created_at=d.created_at,
            updated_at=d.updated_at,
            row_count=d.row_count,
            column_count=d.column_count,
            quality_score=d.quality_score,
            quality_report=d.quality_report
        )
        for d in datasets
    ]

@app.post("/api/v1/datasets", response_model=DatasetResponse)
async def create_dataset(
    dataset_data: DatasetCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    # Validate files exist and belong to user
    for file_id in dataset_data.file_ids:
        result = await db.execute(
            select(FileModel).where(
                (FileModel.id == file_id) & (FileModel.owner_id == current_user.id)
            )
        )
        if not result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail=f"File {file_id} not found")
    
    # Create dataset
    dataset = Dataset(
        id=str(uuid.uuid4()),
        name=dataset_data.name,
        description=dataset_data.description,
        owner_id=current_user.id
    )
    db.add(dataset)
    
    # Link files to dataset
    for file_id in dataset_data.file_ids:
        dataset_file = DatasetFile(
            dataset_id=dataset.id,
            file_id=file_id
        )
        db.add(dataset_file)
    
    await db.commit()
    
    # Calculate dataset metrics in background
    if claude_service:
        background_tasks.add_task(analyze_dataset_with_ai, dataset.id)
    
    # Reload dataset with relationships
    result = await db.execute(
        select(Dataset)
        .where(Dataset.id == dataset.id)
        .options(selectinload(Dataset.dataset_files).selectinload(DatasetFile.file))
    )
    dataset = result.scalar_one()
    
    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        schema=dataset.schema_json,
        files=[
            FileUploadResponse(
                file_id=df.file.id,
                filename=df.file.filename,
                file_type=df.file.file_type,
                size=df.file.size,
                uploaded_at=df.file.uploaded_at,
                preview=df.file.preview_data
            )
            for df in dataset.dataset_files
        ],
        owner_id=dataset.owner_id,
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
        row_count=dataset.row_count,
        column_count=dataset.column_count,
        quality_score=dataset.quality_score,
        quality_report=dataset.quality_report
    )

@app.get("/api/v1/datasets/{dataset_id}/quality", response_model=QualityReport)
async def get_dataset_quality(
    dataset_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get AI-powered quality report for a dataset"""
    result = await db.execute(
        select(Dataset).where(
            (Dataset.id == dataset_id) & (Dataset.owner_id == current_user.id)
        )
    )
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if not claude_service:
        raise HTTPException(status_code=503, detail="AI service not available")
    
    # Run AI analysis
    analyze_dataset_with_ai.delay(dataset_id)
    
    # Return existing report or placeholder
    if dataset.quality_report:
        return QualityReport(**dataset.quality_report)
    else:
        return QualityReport(
            quality_score=0.0,
            issues=[],
            patterns=[],
            recommendations=["AI analysis in progress. Please check back later."],
            pii_concerns=[],
            schema_suggestions={}
        )

# Pipeline endpoints
@app.get("/api/v1/pipelines", response_model=List[PipelineResponse])
async def list_pipelines(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    result = await db.execute(
        select(Pipeline)
        .where(Pipeline.owner_id == current_user.id)
        .offset(skip)
        .limit(limit)
    )
    pipelines = result.scalars().all()
    
    return [
        PipelineResponse(
            id=p.id,
            name=p.name,
            description=p.description,
            dataset_id=p.dataset_id,
            steps=[PipelineStep(**step) for step in p.steps],
            owner_id=p.owner_id,
            created_at=p.created_at,
            updated_at=p.updated_at,
            last_execution_at=p.last_execution_at,
            status=p.status
        )
        for p in pipelines
    ]

@app.post("/api/v1/pipelines", response_model=PipelineResponse)
async def create_pipeline(
    pipeline_data: PipelineCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    # Validate dataset exists and belongs to user
    result = await db.execute(
        select(Dataset).where(
            (Dataset.id == pipeline_data.dataset_id) & (Dataset.owner_id == current_user.id)
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Create pipeline
    pipeline = Pipeline(
        id=str(uuid.uuid4()),
        name=pipeline_data.name,
        description=pipeline_data.description,
        dataset_id=pipeline_data.dataset_id,
        steps=[step.dict() for step in pipeline_data.steps],
        owner_id=current_user.id
    )
    db.add(pipeline)
    await db.commit()
    await db.refresh(pipeline)
    
    return PipelineResponse(
        id=pipeline.id,
        name=pipeline.name,
        description=pipeline.description,
        dataset_id=pipeline.dataset_id,
        steps=[PipelineStep(**step) for step in pipeline.steps],
        owner_id=pipeline.owner_id,
        created_at=pipeline.created_at,
        updated_at=pipeline.updated_at,
        last_execution_at=pipeline.last_execution_at,
        status=pipeline.status
    )

@app.post("/api/v1/pipelines/{pipeline_id}/execute", response_model=PipelineExecutionResponse)
async def execute_pipeline_endpoint(
    pipeline_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Execute a pipeline asynchronously"""
    result = await db.execute(
        select(Pipeline).where(
            (Pipeline.id == pipeline_id) & (Pipeline.owner_id == current_user.id)
        )
    )
    pipeline = result.scalar_one_or_none()
    
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    # Create execution record
    execution = PipelineExecution(
        id=str(uuid.uuid4()),
        pipeline_id=pipeline_id,
        status="pending",
        started_at=datetime.utcnow()
    )
    db.add(execution)
    await db.commit()
    
    # Queue pipeline execution
    execute_pipeline.delay(pipeline_id, execution.id)
    
    return PipelineExecutionResponse(
        id=execution.id,
        pipeline_id=execution.pipeline_id,
        status=execution.status,
        started_at=execution.started_at,
        completed_at=execution.completed_at,
        error=execution.error,
        results=execution.results,
        metrics=execution.metrics
    )

@app.get("/api/v1/pipelines/{pipeline_id}/executions", response_model=List[PipelineExecutionResponse])
async def list_pipeline_executions(
    pipeline_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """List all executions for a pipeline"""
    # Verify pipeline ownership
    result = await db.execute(
        select(Pipeline).where(
            (Pipeline.id == pipeline_id) & (Pipeline.owner_id == current_user.id)
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    # Get executions
    result = await db.execute(
        select(PipelineExecution)
        .where(PipelineExecution.pipeline_id == pipeline_id)
        .order_by(PipelineExecution.started_at.desc())
    )
    executions = result.scalars().all()
    
    return [
        PipelineExecutionResponse(
            id=e.id,
            pipeline_id=e.pipeline_id,
            status=e.status,
            started_at=e.started_at,
            completed_at=e.completed_at,
            error=e.error,
            results=e.results,
            metrics=e.metrics
        )
        for e in executions
    ]

# Serve uploaded files (for development only)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)