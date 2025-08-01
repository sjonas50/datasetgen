from fastapi import FastAPI, HTTPException, Depends, status, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import jwt
import bcrypt
import uuid
import json
import os

# Configuration
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# In-memory storage (for demo purposes)
users_db = {}
datasets_db = {}
pipelines_db = {}

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

class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    schema: Optional[Dict[str, Any]] = None

class DatasetResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    schema: Optional[Dict[str, Any]]
    owner_id: str
    created_at: datetime
    updated_at: datetime

class PipelineCreate(BaseModel):
    name: str
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

class PipelineResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    config: Optional[Dict[str, Any]]
    owner_id: str
    created_at: datetime
    updated_at: datetime

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
)

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
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None or user_id not in users_db:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return users_db[user_id]

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

@app.get("/api/v1/test")
async def test_endpoint():
    return {
        "message": "API is working!",
        "anthropic_key_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
        "database_url_configured": bool(os.getenv("DATABASE_URL"))
    }

# Auth endpoints
@app.post("/api/v1/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate):
    # Check if user exists
    for user in users_db.values():
        if user["email"] == user_data.email or user["username"] == user_data.username:
            raise HTTPException(status_code=400, detail="User already exists")
    
    # Create user
    user_id = str(uuid.uuid4())
    now = datetime.utcnow()
    user = {
        "id": user_id,
        "email": user_data.email,
        "username": user_data.username,
        "hashed_password": get_password_hash(user_data.password),
        "is_active": True,
        "created_at": now
    }
    users_db[user_id] = user
    
    return UserResponse(
        id=user["id"],
        email=user["email"],
        username=user["username"],
        is_active=user["is_active"],
        created_at=user["created_at"]
    )

@app.post("/api/v1/auth/login", response_model=Token)
async def login(username: str = Form(...), password: str = Form(...)):
    # Find user
    user = None
    for u in users_db.values():
        if u["email"] == username or u["username"] == username:
            user = u
            break
    
    if not user or not verify_password(password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create token
    access_token = create_access_token(data={"sub": user["id"]})
    return Token(access_token=access_token)

@app.get("/api/v1/auth/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    return UserResponse(
        id=current_user["id"],
        email=current_user["email"],
        username=current_user["username"],
        is_active=current_user["is_active"],
        created_at=current_user["created_at"]
    )

# Dataset endpoints
@app.get("/api/v1/datasets", response_model=List[DatasetResponse])
async def list_datasets(
    skip: int = 0,
    limit: int = 100,
    current_user: dict = Depends(get_current_user)
):
    user_datasets = [
        d for d in datasets_db.values() 
        if d["owner_id"] == current_user["id"]
    ]
    
    return [
        DatasetResponse(**d)
        for d in user_datasets[skip:skip+limit]
    ]

@app.post("/api/v1/datasets", response_model=DatasetResponse)
async def create_dataset(
    dataset_data: DatasetCreate,
    current_user: dict = Depends(get_current_user)
):
    dataset_id = str(uuid.uuid4())
    now = datetime.utcnow()
    dataset = {
        "id": dataset_id,
        "name": dataset_data.name,
        "description": dataset_data.description,
        "schema": dataset_data.schema,
        "owner_id": current_user["id"],
        "created_at": now,
        "updated_at": now
    }
    datasets_db[dataset_id] = dataset
    
    return DatasetResponse(**dataset)

@app.get("/api/v1/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    current_user: dict = Depends(get_current_user)
):
    dataset = datasets_db.get(dataset_id)
    if not dataset or dataset["owner_id"] != current_user["id"]:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return DatasetResponse(**dataset)

# Pipeline endpoints
@app.get("/api/v1/pipelines", response_model=List[PipelineResponse])
async def list_pipelines(
    skip: int = 0,
    limit: int = 100,
    current_user: dict = Depends(get_current_user)
):
    user_pipelines = [
        p for p in pipelines_db.values() 
        if p["owner_id"] == current_user["id"]
    ]
    
    return [
        PipelineResponse(**p)
        for p in user_pipelines[skip:skip+limit]
    ]

@app.post("/api/v1/pipelines", response_model=PipelineResponse)
async def create_pipeline(
    pipeline_data: PipelineCreate,
    current_user: dict = Depends(get_current_user)
):
    pipeline_id = str(uuid.uuid4())
    now = datetime.utcnow()
    pipeline = {
        "id": pipeline_id,
        "name": pipeline_data.name,
        "description": pipeline_data.description,
        "config": pipeline_data.config,
        "owner_id": current_user["id"],
        "created_at": now,
        "updated_at": now
    }
    pipelines_db[pipeline_id] = pipeline
    
    return PipelineResponse(**pipeline)