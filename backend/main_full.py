from fastapi import FastAPI, HTTPException, Depends, status, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import jwt
import bcrypt
import uuid
import json
import os
import shutil
from pathlib import Path
import pandas as pd
import aiofiles
import mimetypes
from dotenv import load_dotenv

# Load environment variables from root directory
load_dotenv(Path(__file__).parent.parent / '.env')

# Import pipeline executor
from services.pipeline_executor import execute_pipeline, get_execution_status, get_pipeline_executions
from services.pipeline_analyzer import pipeline_analyzer
from services.results_service import results_service

# Import SQLite database
from database_sqlite import db

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory storage for executions (will be replaced with database)
# TODO: Move executions to SQLite for full persistence
executions_db = {}

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
    schema_config: Optional[Dict[str, Any]] = Field(None, alias="schema")

class DatasetResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    schema_config: Optional[Dict[str, Any]] = Field(None, alias="schema")
    files: List[FileUploadResponse]
    owner_id: str
    created_at: datetime
    updated_at: datetime
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    quality_score: Optional[float] = None

class PipelineStep(BaseModel):
    type: str  # e.g., "quality_validation", "pii_detection", "data_cleaning"
    config: Dict[str, Any]

class PipelineCreate(BaseModel):
    name: str
    description: Optional[str] = None
    dataset_id: str
    steps: List[PipelineStep]
    auto_analyze: Optional[bool] = False  # Flag to trigger auto-analysis

class PipelineResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    dataset_id: str
    steps: List[PipelineStep]
    owner_id: str
    created_at: datetime
    updated_at: datetime
    last_execution: Optional[datetime] = None
    status: str = "draft"  # draft, active, paused

class PipelineExecutionResponse(BaseModel):
    id: str
    pipeline_id: str
    status: str  # pending, running, completed, failed
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    results: Optional[Dict[str, Any]] = None

class QualityReport(BaseModel):
    overall_score: float
    column_scores: Dict[str, float]
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    statistics: Dict[str, Any]

# FastAPI app
app = FastAPI(
    title="DatasetGen API",
    description="AI-First Dataset Creation Platform",
    version="1.0.0"
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

# Add OPTIONS handler for file upload
@app.options("/api/v1/files/upload")
async def file_upload_options():
    return {"message": "OK"}

# Security
security = HTTPBearer()

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

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    try:
        print(f"Auth header received: {credentials.credentials[:20]}..." if credentials else "No auth header")
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            print("No user_id in token payload")
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.PyJWTError as e:
        print(f"JWT decode error: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.get_user_by_id(user_id)
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
            preview = {
                "type": "document",
                "format": "pdf",
                "message": "PDF document ready for AI processing",
                "file_size": file_path.stat().st_size
            }
        elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
            preview = {
                "type": "document",
                "format": "word",
                "message": "Word document ready for AI processing",
                "file_size": file_path.stat().st_size
            }
        elif file_type in ["text/plain", "text/markdown", "application/rtf"]:
            # Read text content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            preview = {
                "type": "document",
                "format": "text",
                "content_preview": content[:500] + "..." if len(content) > 500 else content,
                "total_length": len(content),
                "message": "Text document ready for processing"
            }
        elif file_type in ["image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"]:
            preview = {
                "type": "image",
                "format": file_type.split('/')[-1],
                "message": "Image ready for AI vision processing",
                "file_size": file_path.stat().st_size
            }
    except Exception as e:
        preview = {"error": str(e)}
    
    return preview

def calculate_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate basic quality score for a dataframe"""
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    # Basic quality score calculation
    missing_penalty = (missing_cells / total_cells) * 30 if total_cells > 0 else 0
    duplicate_penalty = (duplicate_rows / len(df)) * 20 if len(df) > 0 else 0
    quality_score = max(0, 100 - missing_penalty - duplicate_penalty)
    
    # Column-wise scores
    column_scores = {}
    for col in df.columns:
        col_missing = df[col].isnull().sum()
        col_unique = df[col].nunique()
        col_score = 100
        
        # Penalize missing values
        if len(df) > 0:
            col_score -= (col_missing / len(df)) * 50
        
        # Penalize low cardinality for non-boolean columns
        if col_unique == 1 and len(df) > 1:
            col_score -= 30
            
        column_scores[col] = round(col_score, 2)
    
    # Identify issues
    issues = []
    if missing_cells > 0:
        issues.append({
            "type": "missing_values",
            "severity": "high" if missing_cells / total_cells > 0.1 else "medium",
            "count": missing_cells,
            "columns": df.columns[df.isnull().any()].tolist()
        })
    
    if duplicate_rows > 0:
        issues.append({
            "type": "duplicate_rows",
            "severity": "medium",
            "count": duplicate_rows
        })
    
    # Generate recommendations
    recommendations = []
    if missing_cells > 0:
        recommendations.append("Consider handling missing values through imputation or removal")
    if duplicate_rows > 0:
        recommendations.append("Remove duplicate rows to improve data quality")
    
    return {
        "overall_score": round(quality_score, 2),
        "column_scores": column_scores,
        "issues": issues,
        "recommendations": recommendations,
        "statistics": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_cells": missing_cells,
            "duplicate_rows": duplicate_rows,
            "memory_usage": df.memory_usage(deep=True).sum()
        }
    }

# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to DatasetGen API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "DatasetGen Backend",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/v1/test-dataset-generation")
async def test_dataset_generation(
    dataset_type: str = "qa_pairs",
    current_user: dict = Depends(get_current_user)
):
    """Test endpoint for dataset generation"""
    from services.dataset_generator import dataset_generator
    
    # Create test data
    test_df = pd.DataFrame({
        'text_content': [
            """This is a scanned PDF document with 7 pages.
            Document Information:
            - Type: Scanned/Image-based PDF
            - Pages: 7
            
            The document contains business information and technical specifications
            that would be valuable for training machine learning models."""
        ],
        'source_file': ['test.pdf']
    })
    
    print(f"[TEST] Starting dataset generation test")
    print(f"[TEST] Input DataFrame shape: {test_df.shape}")
    print(f"[TEST] Dataset type: {dataset_type}")
    
    result = await dataset_generator.generate_dataset(
        test_df,
        dataset_type,
        {'min_examples': 5}
    )
    
    print(f"[TEST] Generation result: {result}")
    
    return result

# Auth endpoints
@app.post("/api/v1/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate):
    # Check if user exists
    existing_user = db.get_user_by_username_or_email(user_data.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")
    
    # Create user
    user = db.create_user({
        "email": user_data.email,
        "username": user_data.username,
        "hashed_password": get_password_hash(user_data.password)
    })
    
    return UserResponse(
        id=user["id"],
        email=user["email"],
        username=user["username"],
        is_active=user.get("is_active", True),
        created_at=user["created_at"]
    )

@app.post("/api/v1/auth/login", response_model=Token)
async def login(username: str = Form(...), password: str = Form(...)):
    # Find user
    user = db.get_user_by_username_or_email(username)
    
    if not user or not verify_password(password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create token
    access_token = create_access_token(data={"sub": user["id"]})
    print(f"Login successful for user: {username}, token: {access_token[:20]}...")
    return Token(access_token=access_token)

# File upload endpoints
@app.post("/api/v1/files/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload a file for processing"""
    print(f"File upload request received: {file.filename}, user: {current_user.get('username', 'unknown')}")
    # Validate file type
    allowed_types = [
        # Structured data
        "text/csv", "application/csv",
        "application/json",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        # Documents
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
        "application/msword",  # .doc
        "text/plain",  # .txt
        "text/markdown",  # .md
        "application/rtf",  # .rtf
        # Images
        "image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp",
        # Other
        "application/octet-stream"  # For files with unknown mime types
    ]
    
    file_type = file.content_type or mimetypes.guess_type(file.filename)[0]
    if file_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_type} not supported. Allowed types: CSV, JSON, Excel, PDF, Word (DOC/DOCX), Text, Markdown, RTF, Images (PNG/JPG/GIF/WebP)"
        )
    
    # Create user directory
    user_dir = UPLOAD_DIR / current_user["id"]
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
    
    # Store file metadata in SQLite
    file_data = db.create_file({
        "id": file_id,
        "filename": file.filename,
        "file_type": file_type,
        "file_path": str(file_path),
        "size": len(content),
        "preview_data": preview,
        "owner_id": current_user["id"],
        "uploaded_at": datetime.utcnow()
    })
    
    return FileUploadResponse(
        file_id=file_data["id"],
        filename=file_data["filename"],
        file_type=file_data["file_type"],
        size=file_data["size"],
        uploaded_at=file_data["uploaded_at"],
        preview=file_data.get("preview_data", preview)
    )

@app.get("/api/v1/files", response_model=List[FileUploadResponse])
async def list_files(
    current_user: dict = Depends(get_current_user)
):
    """List all uploaded files for the current user"""
    files = db.get_files_by_owner(current_user["id"])
    
    user_files = [
        FileUploadResponse(
            file_id=f["id"],
            filename=f["filename"],
            file_type=f["file_type"],
            size=f["size"],
            uploaded_at=datetime.fromisoformat(f["uploaded_at"]) if isinstance(f["uploaded_at"], str) else f["uploaded_at"],
            preview=f.get("preview_data")
        )
        for f in files
    ]
    return user_files

@app.post("/api/v1/estimate-cost")
async def estimate_cost_from_files(
    request: dict,
    current_user: dict = Depends(get_current_user)
):
    """Estimate cost for dataset generation from uploaded files without creating a dataset"""
    from services.token_estimator import token_estimator
    
    try:
        # Extract parameters
        file_ids = request.get("file_ids", [])
        dataset_type = request.get("dataset_type", "qa_pairs")
        target_rows = request.get("target_rows")
        custom_instructions = request.get("custom_instructions")
        processing_strategy = request.get("processing_strategy", "auto")
        
        if not file_ids:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Get files from database
        files = db.get_files_by_owner(current_user["id"])
        selected_files = [f for f in files if f["id"] in file_ids]
        
        if len(selected_files) != len(file_ids):
            raise HTTPException(status_code=404, detail="One or more files not found")
        
        # Extract content from files
        all_content = []
        documents = []
        
        for file_info in selected_files:
            file_path = Path(file_info["file_path"])
            
            if file_path.exists():
                # Extract content based on file type
                if file_info["file_type"] in ["text/plain", "text/markdown"]:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                elif file_info["file_type"] == "application/pdf":
                    # For PDFs, use document extractor
                    extraction_result = await document_extractor.extract_from_pdf(file_path, {"use_ai_extraction": False})
                    content = extraction_result.get("text_content", "") or extraction_result.get("enhanced_content", "")
                elif file_info["file_type"] in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                    # For Word docs
                    extraction_result = await document_extractor.extract_from_docx(file_path)
                    content = extraction_result.get("text_content", "")
                else:
                    # For CSV/JSON, just read as text for token counting
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                
                all_content.append(content)
                documents.append({
                    "filename": file_info["filename"],
                    "content": content,
                    "file_type": file_info["file_type"]
                })
        
        # Combine all content
        combined_content = "\n\n---\n\n".join(all_content)
        
        # Estimate cost
        if len(documents) > 1:
            # Multi-document estimation
            cost_estimate = await token_estimator.estimate_multi_document_cost(
                documents,
                dataset_type,
                processing_strategy
            )
        else:
            # Single document estimation
            cost_estimate = await token_estimator.estimate_cost(
                combined_content,
                dataset_type,
                target_rows,
                custom_instructions
            )
        
        return cost_estimate
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error estimating cost: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Cost estimation failed: {str(e)}")

# Dataset endpoints with file support
@app.get("/api/v1/datasets", response_model=List[DatasetResponse])
async def list_datasets(
    skip: int = 0,
    limit: int = 100,
    current_user: dict = Depends(get_current_user)
):
    datasets = db.get_datasets_by_owner(current_user["id"])
    
    # Convert to response models
    response = []
    for dataset in datasets[skip:skip+limit]:
        files = [
            FileUploadResponse(
                file_id=f["id"],
                filename=f["filename"],
                file_type=f["file_type"],
                size=f["size"],
                uploaded_at=datetime.fromisoformat(f["uploaded_at"]) if isinstance(f["uploaded_at"], str) else f["uploaded_at"],
                preview=f.get("preview_data")
            )
            for f in dataset.get("files", [])
        ]
        
        response.append(DatasetResponse(
            id=dataset["id"],
            name=dataset["name"],
            description=dataset["description"],
            schema_config=dataset.get("schema_config"),
            files=files,
            owner_id=dataset["owner_id"],
            created_at=datetime.fromisoformat(dataset["created_at"]) if isinstance(dataset["created_at"], str) else dataset["created_at"],
            updated_at=datetime.fromisoformat(dataset["updated_at"]) if isinstance(dataset["updated_at"], str) else dataset["updated_at"],
            row_count=dataset.get("row_count"),
            column_count=dataset.get("column_count"),
            quality_score=dataset.get("quality_score")
        ))
    
    return response

@app.post("/api/v1/datasets", response_model=DatasetResponse)
async def create_dataset(
    dataset_data: DatasetCreate,
    current_user: dict = Depends(get_current_user)
):
    # Validate files exist and belong to user
    for file_id in dataset_data.file_ids:
        file_info = db.get_file_by_id(file_id, current_user["id"])
        if not file_info:
            raise HTTPException(status_code=404, detail=f"File {file_id} not found or access denied")
    
    # Create dataset
    dataset_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    # Calculate dataset metrics from first file (if CSV/Excel)
    row_count = None
    column_count = None
    quality_score = None
    
    if dataset_data.file_ids:
        first_file = db.get_file_by_id(dataset_data.file_ids[0], current_user["id"])
        if first_file and first_file["file_type"] in ["text/csv", "application/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            try:
                file_path = Path(first_file["file_path"])
                if first_file["file_type"] in ["text/csv", "application/csv"]:
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                
                row_count = len(df)
                column_count = len(df.columns)
                
                # Calculate quality score
                quality_report = calculate_quality_score(df)
                quality_score = quality_report["overall_score"]
            except Exception as e:
                print(f"Error processing file for metrics: {e}")
    
    # Create dataset in SQLite
    dataset = db.create_dataset({
        "id": dataset_id,
        "name": dataset_data.name,
        "description": dataset_data.description,
        "schema_config": dataset_data.schema_config,
        "file_ids": dataset_data.file_ids,
        "owner_id": current_user["id"],
        "created_at": now,
        "updated_at": now,
        "row_count": row_count,
        "column_count": column_count,
        "quality_score": quality_score
    })
    
    # Get files for response
    files = []
    for file_id in dataset_data.file_ids:
        f = db.get_file_by_id(file_id, current_user["id"])
        if f:
            files.append(FileUploadResponse(
                file_id=f["id"],
                filename=f["filename"],
                file_type=f["file_type"],
                size=f["size"],
                uploaded_at=datetime.fromisoformat(f["uploaded_at"]) if isinstance(f["uploaded_at"], str) else f["uploaded_at"],
                preview=f.get("preview_data")
            ))
    
    return DatasetResponse(
        id=dataset["id"],
        name=dataset["name"],
        description=dataset["description"],
        schema_config=dataset["schema_config"],
        files=files,
        owner_id=dataset["owner_id"],
        created_at=dataset["created_at"],
        updated_at=dataset["updated_at"],
        row_count=row_count,
        column_count=column_count,
        quality_score=quality_score
    )

@app.get("/api/v1/datasets/{dataset_id}/quality", response_model=QualityReport)
async def get_dataset_quality(
    dataset_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed quality report for a dataset"""
    # Get dataset from SQLite
    datasets = db.get_datasets_by_owner(current_user["id"])
    dataset = next((d for d in datasets if d["id"] == dataset_id), None)
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Get first CSV/Excel file for analysis
    for file in dataset.get("files", []):
        if file["file_type"] in ["text/csv", "application/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            try:
                file_path = Path(file["file_path"])
                if file["file_type"] in ["text/csv", "application/csv"]:
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                
                quality_report = calculate_quality_score(df)
                return QualityReport(**quality_report)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error analyzing file: {str(e)}")
    
    raise HTTPException(status_code=400, detail="No analyzable files in dataset")

@app.post("/api/v1/datasets/{dataset_id}/analyze-pipeline")
async def analyze_dataset_pipeline(
    dataset_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Use AI to analyze dataset and suggest an appropriate pipeline"""
    # Get dataset from SQLite
    datasets = db.get_datasets_by_owner(current_user["id"])
    dataset = next((d for d in datasets if d["id"] == dataset_id), None)
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        # Analyze dataset with AI
        analysis_result = await pipeline_analyzer.analyze_dataset(dataset)
        
        # Optionally create the pipeline automatically
        if analysis_result.get("confidence_score", 0) > 0.7:
            # Create pipeline in database
            pipeline_config = analysis_result["suggested_pipeline"]
            pipeline_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            pipeline = db.create_pipeline({
                "id": pipeline_id,
                "name": pipeline_config["name"],
                "description": pipeline_config["description"],
                "dataset_id": dataset_id,
                "steps": pipeline_config["steps"],
                "owner_id": current_user["id"],
                "created_at": now,
                "updated_at": now,
                "status": "draft"
            })
            
            analysis_result["created_pipeline_id"] = pipeline_id
        
        return analysis_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/v1/datasets/{dataset_id}/estimate-cost")
async def estimate_dataset_cost(
    dataset_id: str,
    request: dict,
    current_user: dict = Depends(get_current_user)
):
    """Estimate cost for dataset generation"""
    from services.token_estimator import token_estimator
    
    # Get dataset from SQLite
    datasets = db.get_datasets_by_owner(current_user["id"])
    dataset = next((d for d in datasets if d["id"] == dataset_id), None)
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        # Extract parameters
        dataset_type = request.get("dataset_type", "qa_pairs")
        target_rows = request.get("target_rows")
        custom_instructions = request.get("custom_instructions")
        processing_strategy = request.get("processing_strategy", "auto")
        
        # Get all file contents
        all_content = []
        documents = []
        
        for file_info in dataset.get("files", []):
            file_path = Path(file_info["file_path"])
            
            if file_path.exists():
                # Extract content based on file type
                if file_info["file_type"] in ["text/plain", "text/markdown"]:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                elif file_info["file_type"] == "application/pdf":
                    # For PDFs, use document extractor
                    extraction_result = await document_extractor.extract_from_pdf(file_path, {"use_ai_extraction": False})
                    content = extraction_result.get("text_content", "") or extraction_result.get("enhanced_content", "")
                elif file_info["file_type"] in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                    # For Word docs
                    extraction_result = await document_extractor.extract_from_docx(file_path)
                    content = extraction_result.get("text_content", "")
                else:
                    # For CSV/JSON, just read as text for token counting
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                
                all_content.append(content)
                documents.append({
                    "filename": file_info["filename"],
                    "content": content,
                    "file_type": file_info["file_type"]
                })
        
        # Combine all content
        combined_content = "\n\n---\n\n".join(all_content)
        
        # Estimate cost
        if len(documents) > 1:
            # Multi-document estimation
            cost_estimate = await token_estimator.estimate_multi_document_cost(
                documents,
                dataset_type,
                processing_strategy
            )
        else:
            # Single document estimation
            cost_estimate = await token_estimator.estimate_cost(
                combined_content,
                dataset_type,
                target_rows,
                custom_instructions
            )
        
        return cost_estimate
        
    except Exception as e:
        print(f"Error estimating cost: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Cost estimation failed: {str(e)}")

# Pipeline endpoints
@app.get("/api/v1/pipelines", response_model=List[PipelineResponse])
async def list_pipelines(
    skip: int = 0,
    limit: int = 100,
    current_user: dict = Depends(get_current_user)
):
    pipelines = db.get_pipelines_by_owner(current_user["id"])
    
    response = []
    for pipeline in pipelines[skip:skip+limit]:
        response.append(PipelineResponse(
            id=pipeline["id"],
            name=pipeline["name"],
            description=pipeline.get("description"),
            dataset_id=pipeline["dataset_id"],
            steps=[PipelineStep(**step) for step in pipeline.get("steps", [])],
            owner_id=pipeline["owner_id"],
            created_at=datetime.fromisoformat(pipeline["created_at"]) if isinstance(pipeline["created_at"], str) else pipeline["created_at"],
            updated_at=datetime.fromisoformat(pipeline["updated_at"]) if isinstance(pipeline["updated_at"], str) else pipeline["updated_at"],
            last_execution=datetime.fromisoformat(pipeline["last_execution"]) if pipeline.get("last_execution") and isinstance(pipeline["last_execution"], str) else pipeline.get("last_execution"),
            status=pipeline.get("status", "draft")
        ))
    
    return response

@app.post("/api/v1/pipelines", response_model=PipelineResponse)
async def create_pipeline(
    pipeline_data: PipelineCreate,
    current_user: dict = Depends(get_current_user)
):
    # Validate dataset exists and belongs to user
    datasets = db.get_datasets_by_owner(current_user["id"])
    dataset = next((d for d in datasets if d["id"] == pipeline_data.dataset_id), None)
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found or access denied")
    
    # Create pipeline in SQLite
    pipeline_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    pipeline = db.create_pipeline({
        "id": pipeline_id,
        "name": pipeline_data.name,
        "description": pipeline_data.description,
        "dataset_id": pipeline_data.dataset_id,
        "steps": [step.dict() for step in pipeline_data.steps],
        "owner_id": current_user["id"],
        "created_at": now,
        "updated_at": now,
        "status": "draft"
    })
    
    return PipelineResponse(
        id=pipeline["id"],
        name=pipeline["name"],
        description=pipeline.get("description"),
        dataset_id=pipeline["dataset_id"],
        steps=[PipelineStep(**step) for step in pipeline.get("steps", [])],
        owner_id=pipeline["owner_id"],
        created_at=pipeline["created_at"],
        updated_at=pipeline["updated_at"],
        last_execution=pipeline.get("last_execution"),
        status=pipeline.get("status", "draft")
    )

@app.post("/api/v1/pipelines/{pipeline_id}/execute", response_model=PipelineExecutionResponse)
async def execute_pipeline_endpoint(
    pipeline_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Execute a pipeline"""
    # Get pipeline from SQLite
    pipelines = db.get_pipelines_by_owner(current_user["id"])
    pipeline = next((p for p in pipelines if p["id"] == pipeline_id), None)
    
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    # Get the dataset from SQLite
    datasets = db.get_datasets_by_owner(current_user["id"])
    dataset = next((d for d in datasets if d["id"] == pipeline["dataset_id"]), None)
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Execute pipeline asynchronously
    execution_id = execute_pipeline(pipeline, dataset)
    
    # Get initial status
    execution = get_execution_status(execution_id)
    if not execution:
        raise HTTPException(status_code=500, detail="Failed to start pipeline execution")
    
    # Store execution in memory db
    executions_db[execution_id] = execution
    
    # Update pipeline last execution in SQLite
    db.update_pipeline(pipeline_id, {
        "last_execution": datetime.utcnow(),
        "status": "active"
    })
    
    return PipelineExecutionResponse(
        id=execution["id"],
        pipeline_id=execution["pipeline_id"],
        status=execution["status"],
        started_at=execution["started_at"],
        completed_at=execution.get("completed_at"),
        error=execution.get("error"),
        results=execution.get("results")
    )

@app.get("/api/v1/pipelines/{pipeline_id}/executions", response_model=List[PipelineExecutionResponse])
async def list_pipeline_executions(
    pipeline_id: str,
    current_user: dict = Depends(get_current_user)
):
    """List all executions for a pipeline"""
    # Get pipeline from SQLite
    pipelines = db.get_pipelines_by_owner(current_user["id"])
    pipeline = next((p for p in pipelines if p["id"] == pipeline_id), None)
    
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    # Get executions from the execution tracker
    pipeline_executions = get_pipeline_executions(pipeline_id)
    
    # Convert to response models, handling numpy types
    executions = []
    for e in pipeline_executions:
        # Convert datetime strings to datetime objects if needed
        started_at = e["started_at"]
        if isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at)
        
        completed_at = e.get("completed_at")
        if completed_at and isinstance(completed_at, str):
            completed_at = datetime.fromisoformat(completed_at)
        
        # Convert results to ensure no numpy types
        results = e.get("results")
        if results:
            # Convert numpy types to Python types
            import numpy as np
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                return obj
            
            results = convert_numpy(results)
        
        executions.append(PipelineExecutionResponse(
            id=e["id"],
            pipeline_id=e["pipeline_id"],
            status=e["status"],
            started_at=started_at,
            completed_at=completed_at,
            error=e.get("error"),
            results=results
        ))
    
    return sorted(executions, key=lambda x: x.started_at, reverse=True)

@app.get("/api/v1/pipelines/{pipeline_id}/executions/{execution_id}/results")
async def get_execution_results(
    pipeline_id: str,
    execution_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed results for a specific execution"""
    # Verify pipeline ownership
    pipelines = db.get_pipelines_by_owner(current_user["id"])
    pipeline = next((p for p in pipelines if p["id"] == pipeline_id), None)
    
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    # Get execution status
    execution = get_execution_status(execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    # Get results from results service
    results = results_service.get_execution_results(execution_id)
    
    return {
        "execution": execution,
        "results": results
    }

@app.get("/api/v1/pipelines/{pipeline_id}/executions/{execution_id}/download")
async def download_execution_results(
    pipeline_id: str,
    execution_id: str,
    format: str = "csv",
    current_user: dict = Depends(get_current_user)
):
    """Download execution results in different formats"""
    from fastapi.responses import FileResponse
    
    # Verify pipeline ownership
    pipelines = db.get_pipelines_by_owner(current_user["id"])
    pipeline = next((p for p in pipelines if p["id"] == pipeline_id), None)
    
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    # Get execution
    execution = get_execution_status(execution_id)
    if not execution or not execution.get("metrics", {}).get("output_path"):
        raise HTTPException(status_code=404, detail="No output file found")
    
    output_path = Path(execution["metrics"]["output_path"])
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    # Convert format if needed
    if format != "csv":
        import pandas as pd
        df = pd.read_csv(output_path)
        
        if format == "json":
            json_path = output_path.with_suffix(".json")
            df.to_json(json_path, orient="records", indent=2)
            return FileResponse(
                path=str(json_path),
                filename=f"results_{execution_id}.json",
                media_type="application/json"
            )
        elif format == "excel":
            excel_path = output_path.with_suffix(".xlsx")
            df.to_excel(excel_path, index=False)
            return FileResponse(
                path=str(excel_path),
                filename=f"results_{execution_id}.xlsx",
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # Default to CSV
    return FileResponse(
        path=str(output_path),
        filename=f"results_{execution_id}.csv",
        media_type="text/csv"
    )

# Serve uploaded files (for development only)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)