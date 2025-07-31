from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
from typing import Dict, Any

from core.config import settings
from core.database import engine, Base
from api.v1.router import api_router
from core.logging import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle events
    """
    # Startup
    logger.info("Starting up Dataset Creation Platform API...")
    
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Database initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    await engine.dispose()


# Create FastAPI application
app = FastAPI(
    title="AI-First Dataset Creation Platform",
    description="Transform raw enterprise data into high-quality ML training datasets",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, Any])
async def root():
    """
    Root endpoint - API health check
    """
    return {
        "message": "AI-First Dataset Creation Platform API",
        "version": "0.1.0",
        "status": "healthy",
        "docs": "/docs"
    }


@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """
    Detailed health check endpoint
    """
    try:
        # Check database connection
        async with engine.connect() as conn:
            await conn.execute("SELECT 1")
        
        return {
            "status": "healthy",
            "database": "connected",
            "version": "0.1.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")


# Include API routers
app.include_router(api_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )