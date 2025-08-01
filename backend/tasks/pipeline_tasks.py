"""
Pipeline execution tasks
"""

import os
import json
import uuid
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from celery import Task
from sqlalchemy.orm import Session

from celery_app import celery_app
from database import SessionLocal, Pipeline, PipelineExecution, ProcessingLog, Dataset, File, DatasetFile
from services.claude_service import ClaudeService
from tasks.processing_tasks import (
    quality_validation_task,
    pii_detection_task,
    data_cleaning_task,
    outlier_detection_task,
    data_transformation_task,
    schema_validation_task
)

class PipelineTask(Task):
    """Base task with database session management"""
    _db = None
    _claude_service = None
    
    @property
    def db(self) -> Session:
        if self._db is None:
            self._db = SessionLocal()
        return self._db
    
    @property
    def claude_service(self) -> ClaudeService:
        if self._claude_service is None:
            self._claude_service = ClaudeService()
        return self._claude_service
    
    def after_return(self, *args, **kwargs):
        if self._db is not None:
            self._db.close()
            self._db = None

@celery_app.task(base=PipelineTask, bind=True, name="execute_pipeline")
def execute_pipeline(self, pipeline_id: str, execution_id: str) -> Dict[str, Any]:
    """
    Execute a complete pipeline
    """
    try:
        # Get pipeline from database
        pipeline = self.db.query(Pipeline).filter_by(id=pipeline_id).first()
        if not pipeline:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        # Update execution status
        execution = self.db.query(PipelineExecution).filter_by(id=execution_id).first()
        if not execution:
            raise ValueError(f"Execution {execution_id} not found")
        
        execution.status = "running"
        execution.started_at = datetime.utcnow()
        self.db.commit()
        
        # Load dataset
        dataset = pipeline.dataset
        df = load_dataset_files(self.db, dataset)
        
        # Execute each step
        results = {}
        step_outputs = {}
        current_df = df.copy()
        
        for i, step in enumerate(pipeline.steps):
            step_name = f"step_{i}_{step['type']}"
            
            # Create processing log
            log = ProcessingLog(
                id=str(uuid.uuid4()),
                execution_id=execution_id,
                step_name=step_name,
                status="running",
                started_at=datetime.utcnow(),
                input_records=len(current_df)
            )
            self.db.add(log)
            self.db.commit()
            
            try:
                # Execute step based on type
                if step['type'] == 'quality_validation':
                    result = quality_validation_task(current_df, step['config'])
                    step_outputs[step_name] = result
                    
                elif step['type'] == 'pii_detection':
                    result = pii_detection_task(current_df, step['config'])
                    current_df = result['processed_df']
                    step_outputs[step_name] = result['report']
                    
                elif step['type'] == 'data_cleaning':
                    result = data_cleaning_task(current_df, step['config'])
                    current_df = result['cleaned_df']
                    step_outputs[step_name] = result['operations']
                    
                elif step['type'] == 'outlier_detection':
                    result = outlier_detection_task(current_df, step['config'])
                    current_df = result['processed_df']
                    step_outputs[step_name] = result['outliers']
                    
                elif step['type'] == 'data_transformation':
                    # Use Claude for intelligent transformation
                    result = data_transformation_task.delay(
                        current_df.to_dict(orient='records'),
                        step['config']
                    ).get()
                    current_df = pd.DataFrame(result['transformed_data'])
                    step_outputs[step_name] = result['transformation_log']
                    
                elif step['type'] == 'schema_validation':
                    result = schema_validation_task(current_df, step['config'])
                    step_outputs[step_name] = result
                
                # Update log
                log.status = "completed"
                log.completed_at = datetime.utcnow()
                log.output_records = len(current_df)
                log.metrics = step_outputs.get(step_name, {})
                self.db.commit()
                
            except Exception as e:
                # Log error
                log.status = "failed"
                log.completed_at = datetime.utcnow()
                log.errors = [str(e)]
                self.db.commit()
                raise
        
        # Save processed dataset
        output_dataset = create_output_dataset(self.db, current_df, pipeline, execution_id)
        
        # Update execution
        execution.status = "completed"
        execution.completed_at = datetime.utcnow()
        execution.results = step_outputs
        execution.output_dataset_id = output_dataset.id
        execution.metrics = {
            "input_records": len(df),
            "output_records": len(current_df),
            "duration_seconds": (execution.completed_at - execution.started_at).total_seconds(),
            "steps_completed": len(pipeline.steps)
        }
        
        # Update pipeline last execution
        pipeline.last_execution_at = datetime.utcnow()
        pipeline.status = "active"
        
        self.db.commit()
        
        return {
            "status": "completed",
            "execution_id": execution_id,
            "output_dataset_id": output_dataset.id,
            "metrics": execution.metrics
        }
        
    except Exception as e:
        # Update execution with error
        execution = self.db.query(PipelineExecution).filter_by(id=execution_id).first()
        if execution:
            execution.status = "failed"
            execution.completed_at = datetime.utcnow()
            execution.error = str(e)
            self.db.commit()
        
        raise

def load_dataset_files(db: Session, dataset: Dataset) -> pd.DataFrame:
    """Load all files from a dataset into a single DataFrame"""
    dfs = []
    
    for dataset_file in dataset.dataset_files:
        file = dataset_file.file
        file_path = Path(file.file_path)
        
        if file.file_type in ["text/csv", "application/csv"]:
            df = pd.read_csv(file_path)
        elif file.file_type == "application/json":
            df = pd.read_json(file_path)
        elif file.file_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
            df = pd.read_excel(file_path)
        else:
            continue  # Skip unsupported file types
        
        dfs.append(df)
    
    if not dfs:
        raise ValueError("No valid data files found in dataset")
    
    # Combine dataframes
    if len(dfs) == 1:
        return dfs[0]
    else:
        # Try to concatenate with common columns
        return pd.concat(dfs, ignore_index=True, sort=False)

def create_output_dataset(
    db: Session, 
    df: pd.DataFrame, 
    pipeline: Pipeline, 
    execution_id: str
) -> Dataset:
    """Create a new dataset from pipeline output"""
    # Save DataFrame to file
    output_dir = Path("uploads") / "pipeline_outputs"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{pipeline.name}_{timestamp}.csv"
    output_path = output_dir / output_filename
    
    df.to_csv(output_path, index=False)
    
    # Create file record
    file = File(
        id=str(uuid.uuid4()),
        filename=output_filename,
        file_type="text/csv",
        file_path=str(output_path),
        size=output_path.stat().st_size,
        owner_id=pipeline.owner_id,
        preview_data={
            "rows": len(df),
            "columns": list(df.columns),
            "sample_data": df.head(5).to_dict(orient="records")
        }
    )
    db.add(file)
    
    # Create dataset record
    dataset = Dataset(
        id=str(uuid.uuid4()),
        name=f"{pipeline.name} Output - {timestamp}",
        description=f"Output from pipeline execution {execution_id}",
        row_count=len(df),
        column_count=len(df.columns),
        owner_id=pipeline.owner_id,
        schema_json={
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
    )
    db.add(dataset)
    
    # Link file to dataset
    dataset_file = DatasetFile(
        dataset_id=dataset.id,
        file_id=file.id
    )
    db.add(dataset_file)
    
    db.commit()
    return dataset

@celery_app.task(name="analyze_dataset_with_ai")
def analyze_dataset_with_ai(dataset_id: str) -> Dict[str, Any]:
    """
    Analyze dataset quality using Claude AI
    """
    db = SessionLocal()
    try:
        dataset = db.query(Dataset).filter_by(id=dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Load dataset
        df = load_dataset_files(db, dataset)
        
        # Use Claude for deep analysis
        claude_service = ClaudeService()
        import asyncio
        
        # Run async method in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        analysis = loop.run_until_complete(
            claude_service.analyze_data_quality(df, context=dataset.description or "")
        )
        
        # Update dataset with analysis results
        dataset.quality_score = analysis.get("quality_score", 0)
        dataset.quality_report = analysis
        db.commit()
        
        return analysis
        
    finally:
        db.close()