from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Tuple, Optional
import uuid

from models.dataset import Dataset, DatasetStatus
from schemas.dataset import DatasetCreate, DatasetResponse


class DatasetService:
    """
    Service for dataset operations
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_dataset(self, dataset: DatasetCreate) -> DatasetResponse:
        """
        Create a new dataset
        """
        db_dataset = Dataset(
            name=dataset.name,
            description=dataset.description,
            source_type=dataset.source_type,
            source_config=dataset.source_config,
            status=DatasetStatus.CREATED
        )
        
        self.db.add(db_dataset)
        await self.db.commit()
        await self.db.refresh(db_dataset)
        
        return DatasetResponse.from_orm(db_dataset)
    
    async def list_datasets(
        self, 
        skip: int = 0, 
        limit: int = 10, 
        search: Optional[str] = None
    ) -> Tuple[List[DatasetResponse], int]:
        """
        List datasets with pagination and search
        """
        query = select(Dataset)
        
        if search:
            query = query.where(
                Dataset.name.ilike(f"%{search}%") | 
                Dataset.description.ilike(f"%{search}%")
            )
        
        # Get total count
        count_query = select(func.count()).select_from(Dataset)
        if search:
            count_query = count_query.where(
                Dataset.name.ilike(f"%{search}%") | 
                Dataset.description.ilike(f"%{search}%")
            )
        
        total_result = await self.db.execute(count_query)
        total = total_result.scalar()
        
        # Get paginated results
        query = query.offset(skip).limit(limit)
        result = await self.db.execute(query)
        datasets = result.scalars().all()
        
        return [DatasetResponse.from_orm(ds) for ds in datasets], total
    
    async def get_dataset(self, dataset_id: uuid.UUID) -> Optional[DatasetResponse]:
        """
        Get a dataset by ID
        """
        result = await self.db.execute(
            select(Dataset).where(Dataset.id == dataset_id)
        )
        dataset = result.scalar_one_or_none()
        
        if dataset:
            return DatasetResponse.from_orm(dataset)
        return None
    
    async def upload_data(self, dataset_id: uuid.UUID, file):
        """
        Handle data file upload
        """
        # TODO: Implement file processing
        # - Save file to storage
        # - Extract schema
        # - Calculate statistics
        # - Update dataset record
        
        return {"status": "uploaded", "message": "File processing started"}
    
    async def delete_dataset(self, dataset_id: uuid.UUID) -> bool:
        """
        Delete a dataset
        """
        result = await self.db.execute(
            select(Dataset).where(Dataset.id == dataset_id)
        )
        dataset = result.scalar_one_or_none()
        
        if dataset:
            await self.db.delete(dataset)
            await self.db.commit()
            return True
        
        return False