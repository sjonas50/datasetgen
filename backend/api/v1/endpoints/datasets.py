from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import uuid

from core.database import get_db
from schemas.dataset import DatasetCreate, DatasetResponse, DatasetList
from models.dataset import Dataset
from services.dataset_service import DatasetService

router = APIRouter()


@router.post("/", response_model=DatasetResponse)
async def create_dataset(
    dataset: DatasetCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new dataset
    """
    service = DatasetService(db)
    return await service.create_dataset(dataset)


@router.get("/", response_model=DatasetList)
async def list_datasets(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    List all datasets with pagination
    """
    service = DatasetService(db)
    datasets, total = await service.list_datasets(skip, limit, search)
    
    return DatasetList(
        datasets=datasets,
        total=total,
        skip=skip,
        limit=limit
    )


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific dataset by ID
    """
    service = DatasetService(db)
    dataset = await service.get_dataset(dataset_id)
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return dataset


@router.post("/{dataset_id}/upload")
async def upload_data(
    dataset_id: uuid.UUID,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload data file to dataset
    """
    service = DatasetService(db)
    
    # Validate file type
    if not any(file.filename.endswith(ext) for ext in [".csv", ".json", ".xlsx"]):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Supported: CSV, JSON, XLSX"
        )
    
    result = await service.upload_data(dataset_id, file)
    return result


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a dataset
    """
    service = DatasetService(db)
    success = await service.delete_dataset(dataset_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return {"message": "Dataset deleted successfully"}