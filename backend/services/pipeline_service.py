from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Dict, Any
import uuid

from models.pipeline import Pipeline, PipelineStatus, PipelineExecution
from schemas.pipeline import PipelineCreate, PipelineResponse, PipelineExecute, PipelineUpdate
from services.pipeline import PipelineExecutor, PipelineOptimizer
from services.pipeline.celery_tasks import CeleryPipelineService
from services.pipeline.base import PipelineConfig, StepConfig


class PipelineService:
    """
    Service for pipeline operations
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_pipeline(self, pipeline: PipelineCreate) -> PipelineResponse:
        """
        Create a new pipeline
        """
        # Convert steps to config format
        config = {
            "steps": [step.dict() for step in pipeline.steps],
            "version": "1.0"
        }
        
        db_pipeline = Pipeline(
            name=pipeline.name,
            description=pipeline.description,
            config=config,
            llm_config=pipeline.llm_config,
            status=PipelineStatus.DRAFT
        )
        
        self.db.add(db_pipeline)
        await self.db.commit()
        await self.db.refresh(db_pipeline)
        
        return PipelineResponse.from_orm(db_pipeline)
    
    async def list_pipelines(self, skip: int = 0, limit: int = 10) -> List[PipelineResponse]:
        """
        List all pipelines
        """
        result = await self.db.execute(
            select(Pipeline).offset(skip).limit(limit)
        )
        pipelines = result.scalars().all()
        
        return [PipelineResponse.from_orm(p) for p in pipelines]
    
    async def get_pipeline(self, pipeline_id: uuid.UUID) -> PipelineResponse:
        """
        Get a pipeline by ID
        """
        result = await self.db.execute(
            select(Pipeline).where(Pipeline.id == pipeline_id)
        )
        pipeline = result.scalar_one_or_none()
        
        if pipeline:
            return PipelineResponse.from_orm(pipeline)
        return None
    
    async def execute_pipeline(
        self, 
        pipeline_id: uuid.UUID, 
        execution: PipelineExecute
    ) -> Dict[str, Any]:
        """
        Execute a pipeline using the new execution engine
        """
        # Get pipeline
        result = await self.db.execute(
            select(Pipeline).where(Pipeline.id == pipeline_id)
        )
        pipeline = result.scalar_one_or_none()
        
        if not pipeline:
            raise ValueError("Pipeline not found")
        
        # Convert to PipelineConfig
        pipeline_config = PipelineConfig(
            id=str(pipeline.id),
            name=pipeline.name,
            description=pipeline.description,
            steps=[StepConfig(**step) for step in pipeline.config.get("steps", [])]
        )
        
        # Load initial data if dataset specified
        initial_data = None
        if execution.dataset_id:
            # TODO: Load dataset
            pass
        
        # Execute via Celery for async processing
        if not execution.dry_run:
            task_id = CeleryPipelineService.submit_pipeline(
                pipeline_config,
                initial_data
            )
            
            # Create execution record
            execution_record = PipelineExecution(
                pipeline_id=pipeline_id,
                dataset_id=execution.dataset_id,
                status="running"
            )
            
            self.db.add(execution_record)
            await self.db.commit()
            
            return {
                "execution_id": str(execution_record.id),
                "task_id": task_id,
                "status": "started"
            }
        else:
            # Dry run - just optimize and return plan
            optimizer = PipelineOptimizer()
            optimized = await optimizer.optimize(pipeline_config)
            
            return {
                "execution_id": None,
                "status": "dry_run",
                "original_steps": len(pipeline_config.steps),
                "optimized_steps": len(optimized.steps),
                "optimization_report": optimizer.get_optimization_report()
            }
    
    async def update_pipeline(self, pipeline_id: uuid.UUID, pipeline_update: PipelineUpdate) -> PipelineResponse:
        """
        Update an existing pipeline
        """
        result = await self.db.execute(
            select(Pipeline).where(Pipeline.id == pipeline_id)
        )
        pipeline = result.scalar_one_or_none()
        
        if not pipeline:
            return None
        
        # Update fields if provided
        if pipeline_update.name is not None:
            pipeline.name = pipeline_update.name
        if pipeline_update.description is not None:
            pipeline.description = pipeline_update.description
        if pipeline_update.status is not None:
            pipeline.status = pipeline_update.status
        if pipeline_update.config is not None:
            pipeline.config = pipeline_update.config
        if pipeline_update.llm_config is not None:
            pipeline.llm_config = pipeline_update.llm_config
        
        await self.db.commit()
        await self.db.refresh(pipeline)
        
        return PipelineResponse.from_orm(pipeline)
    
    async def get_pipeline_executions(self, pipeline_id: uuid.UUID) -> List[Dict[str, Any]]:
        """
        Get execution history for a pipeline
        """
        result = await self.db.execute(
            select(PipelineExecution)
            .where(PipelineExecution.pipeline_id == pipeline_id)
            .order_by(PipelineExecution.created_at.desc())
        )
        executions = result.scalars().all()
        
        return [
            {
                "id": e.id,
                "status": e.status,
                "started_at": e.started_at,
                "completed_at": e.completed_at,
                "rows_processed": e.rows_processed,
                "error_message": e.error_message
            }
            for e in executions
        ]
    
    async def delete_pipeline(self, pipeline_id: uuid.UUID) -> bool:
        """
        Delete a pipeline
        """
        result = await self.db.execute(
            select(Pipeline).where(Pipeline.id == pipeline_id)
        )
        pipeline = result.scalar_one_or_none()
        
        if pipeline:
            await self.db.delete(pipeline)
            await self.db.commit()
            return True
        
        return False