from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, DateTime, Boolean, Text, Float, Integer, select
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import jwt
import bcrypt
import os
import uuid
import json

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://platform_user:platform_pass@localhost:5432/dataset_platform")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Database setup
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = async_sessionmaker(engine, expire_on_commit=False)
Base = declarative_base()

# Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False)
    username = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    description = Column(Text)
    schema_json = Column(Text)
    owner_id = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Pipeline(Base):
    __tablename__ = "pipelines"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    description = Column(Text)
    config_json = Column(Text)
    owner_id = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

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

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:80"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Database dependency
async def get_db():
    async with async_session() as session:
        yield session

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
    db: AsyncSession = Depends(get_db)
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

# Auth endpoints
@app.post("/api/v1/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    # Check if user exists
    result = await db.execute(
        select(User).where((User.email == user_data.email) | (User.username == user_data.username))
    )
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="User already exists")
    
    # Create user
    user = User(
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
async def login(username: str, password: str, db: AsyncSession = Depends(get_db)):
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

@app.get("/api/v1/auth/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        is_active=current_user.is_active,
        created_at=current_user.created_at
    )

# Dataset endpoints
@app.get("/api/v1/datasets", response_model=List[DatasetResponse])
async def list_datasets(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(
        select(Dataset)
        .where(Dataset.owner_id == current_user.id)
        .offset(skip)
        .limit(limit)
    )
    datasets = result.scalars().all()
    
    return [
        DatasetResponse(
            id=d.id,
            name=d.name,
            description=d.description,
            schema=json.loads(d.schema_json) if d.schema_json else None,
            owner_id=d.owner_id,
            created_at=d.created_at,
            updated_at=d.updated_at
        )
        for d in datasets
    ]

@app.post("/api/v1/datasets", response_model=DatasetResponse)
async def create_dataset(
    dataset_data: DatasetCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    dataset = Dataset(
        name=dataset_data.name,
        description=dataset_data.description,
        schema_json=json.dumps(dataset_data.schema) if dataset_data.schema else None,
        owner_id=current_user.id
    )
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    
    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        schema=json.loads(dataset.schema_json) if dataset.schema_json else None,
        owner_id=dataset.owner_id,
        created_at=dataset.created_at,
        updated_at=dataset.updated_at
    )

@app.get("/api/v1/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(
        select(Dataset).where(
            (Dataset.id == dataset_id) & (Dataset.owner_id == current_user.id)
        )
    )
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        schema=json.loads(dataset.schema_json) if dataset.schema_json else None,
        owner_id=dataset.owner_id,
        created_at=dataset.created_at,
        updated_at=dataset.updated_at
    )

# Pipeline endpoints
@app.get("/api/v1/pipelines", response_model=List[PipelineResponse])
async def list_pipelines(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
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
            config=json.loads(p.config_json) if p.config_json else None,
            owner_id=p.owner_id,
            created_at=p.created_at,
            updated_at=p.updated_at
        )
        for p in pipelines
    ]

@app.post("/api/v1/pipelines", response_model=PipelineResponse)
async def create_pipeline(
    pipeline_data: PipelineCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    pipeline = Pipeline(
        name=pipeline_data.name,
        description=pipeline_data.description,
        config_json=json.dumps(pipeline_data.config) if pipeline_data.config else None,
        owner_id=current_user.id
    )
    db.add(pipeline)
    await db.commit()
    await db.refresh(pipeline)
    
    return PipelineResponse(
        id=pipeline.id,
        name=pipeline.name,
        description=pipeline.description,
        config=json.loads(pipeline.config_json) if pipeline.config_json else None,
        owner_id=pipeline.owner_id,
        created_at=pipeline.created_at,
        updated_at=pipeline.updated_at
    )

# Initialize database
@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Database tables created")

@app.on_event("shutdown")
async def shutdown():
    await engine.dispose()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)