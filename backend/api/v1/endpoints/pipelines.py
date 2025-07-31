from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import uuid

from core.database import get_db
from schemas.pipeline import PipelineCreate, PipelineResponse, PipelineExecute, PipelineUpdate
from services.pipeline_service import PipelineService

router = APIRouter()


@router.post("/", response_model=PipelineResponse)
async def create_pipeline(
    pipeline: PipelineCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new data processing pipeline
    """
    service = PipelineService(db)
    return await service.create_pipeline(pipeline)


@router.get("/", response_model=List[PipelineResponse])
async def list_pipelines(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """
    List all pipelines
    """
    service = PipelineService(db)
    return await service.list_pipelines(skip, limit)


@router.get("/{pipeline_id}", response_model=PipelineResponse)
async def get_pipeline(
    pipeline_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific pipeline by ID
    """
    service = PipelineService(db)
    pipeline = await service.get_pipeline(pipeline_id)
    
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    return pipeline


@router.post("/{pipeline_id}/execute")
async def execute_pipeline(
    pipeline_id: uuid.UUID,
    execution: PipelineExecute,
    db: AsyncSession = Depends(get_db)
):
    """
    Execute a pipeline with given parameters
    """
    service = PipelineService(db)
    
    try:
        result = await service.execute_pipeline(pipeline_id, execution)
        return {"status": "started", "execution_id": result["execution_id"]}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")


@router.patch("/{pipeline_id}", response_model=PipelineResponse)
async def update_pipeline(
    pipeline_id: uuid.UUID,
    pipeline_update: PipelineUpdate,
    db: AsyncSession = Depends(get_db)
):
    """
    Update a pipeline
    """
    service = PipelineService(db)
    updated_pipeline = await service.update_pipeline(pipeline_id, pipeline_update)
    
    if not updated_pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    return updated_pipeline


@router.get("/{pipeline_id}/executions")
async def get_pipeline_executions(
    pipeline_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get execution history for a pipeline
    """
    service = PipelineService(db)
    executions = await service.get_pipeline_executions(pipeline_id)
    return {"pipeline_id": pipeline_id, "executions": executions}


@router.delete("/{pipeline_id}")
async def delete_pipeline(
    pipeline_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a pipeline
    """
    service = PipelineService(db)
    success = await service.delete_pipeline(pipeline_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    return {"message": "Pipeline deleted successfully"}