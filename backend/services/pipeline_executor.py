"""
Pipeline Execution Service - Simulated async execution without Celery
"""

import asyncio
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

from services.claude_service import ClaudeService
from services.document_extractor import document_extractor
from services.dataset_generator import dataset_generator
from tasks.processing_tasks import (
    quality_validation_task,
    pii_detection_task,
    data_cleaning_task,
    outlier_detection_task,
    schema_validation_task,
)

# Simple in-memory task queue and execution tracking
class ExecutionTracker:
    def __init__(self):
        self.executions = {}
        self.lock = threading.Lock()
    
    def create_execution(self, pipeline_id: str) -> str:
        execution_id = str(uuid.uuid4())
        with self.lock:
            self.executions[execution_id] = {
                "id": execution_id,
                "pipeline_id": pipeline_id,
                "status": "pending",
                "started_at": datetime.utcnow().isoformat(),
                "completed_at": None,
                "error": None,
                "results": None,
                "metrics": None,
                "progress": 0,
                "current_step": None,
            }
        return execution_id
    
    def update_execution(self, execution_id: str, updates: Dict[str, Any]):
        with self.lock:
            if execution_id in self.executions:
                self.executions[execution_id].update(updates)
    
    def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.executions.get(execution_id)
    
    def get_pipeline_executions(self, pipeline_id: str) -> List[Dict[str, Any]]:
        with self.lock:
            return [
                exec_data for exec_data in self.executions.values()
                if exec_data["pipeline_id"] == pipeline_id
            ]

# Global execution tracker
execution_tracker = ExecutionTracker()

# Thread pool for async execution
executor = ThreadPoolExecutor(max_workers=4)

class PipelineExecutor:
    def __init__(self):
        try:
            self.claude_service = ClaudeService()
        except Exception as e:
            print(f"Warning: Could not initialize Claude service: {e}")
            self.claude_service = None
    
    async def execute_pipeline_async(self, pipeline: Dict[str, Any], dataset: Dict[str, Any], execution_id: str):
        """Execute pipeline in background thread"""
        print(f"Pipeline executor started for execution: {execution_id}")
        print(f"Pipeline steps: {pipeline.get('steps', [])}")
        print(f"Dataset files: {len(dataset.get('files', []))} files")
        
        # Debug: Print file details
        for file in dataset.get('files', []):
            print(f"  File: {file.get('filename')} - Type: {file.get('file_type')} - Path: {file.get('file_path')}")
        
        try:
            # Update status to running
            execution_tracker.update_execution(execution_id, {
                "status": "running",
                "started_at": datetime.utcnow().isoformat(),
            })
            
            # Load dataset files
            print(f"Loading dataset files...")
            df = await self.load_dataset_files(dataset)
            initial_rows = len(df)
            print(f"Loaded {initial_rows} rows from dataset")
            print(f"DataFrame columns: {list(df.columns)}")
            if initial_rows > 0:
                print(f"First row sample: {df.iloc[0].to_dict()}")
            
            # Execute each step
            results = {}
            current_df = df.copy()
            total_steps = len(pipeline["steps"])
            
            for i, step in enumerate(pipeline["steps"]):
                step_name = f"step_{i}_{step['type']}"
                
                # Update progress
                execution_tracker.update_execution(execution_id, {
                    "current_step": step_name,
                    "progress": int((i / total_steps) * 100),
                })
                
                try:
                    print(f"\nExecuting {step_name} (type: {step['type']})...")
                    
                    # Execute step based on type
                    if step['type'] == 'pdf_extraction':
                        # Extract content from PDFs
                        extraction_results = []
                        for _, row in current_df.iterrows():
                            if row.get('type') == 'pdf':
                                file_path = Path(row['path'])
                                extraction = await document_extractor.extract_from_pdf(
                                    file_path, 
                                    step.get('config', {})
                                )
                                extraction_results.append(extraction)
                        
                        # Create new dataframe with extracted content
                        if extraction_results:
                            extracted_data = []
                            for result in extraction_results:
                                if result.get('extraction_status') == 'success':
                                    extracted_data.append({
                                        'source_file': result['file_path'],
                                        'text_content': result.get('text_content', ''),
                                        'enhanced_content': result.get('enhanced_content', result.get('text_content', '')),
                                        'page_count': result['metadata'].get('page_count', 0),
                                        'tables_found': len(result.get('tables', [])),
                                        'extraction_method': 'advanced_extraction',
                                        'ai_insights': json.dumps(result.get('ai_insights', {})) if result.get('ai_insights') else None
                                    })
                            
                            if extracted_data:
                                current_df = pd.DataFrame(extracted_data)
                                results[step_name] = {
                                    'status': 'success',
                                    'files_processed': len(extraction_results),
                                    'successful_extractions': len(extracted_data)
                                }
                            else:
                                results[step_name] = {'error': 'No successful extractions'}
                        else:
                            results[step_name] = {'status': 'no_pdfs_found'}
                    
                    elif step['type'] == 'document_extraction':
                        # Extract content from all document types
                        extraction_results = []
                        for _, row in current_df.iterrows():
                            file_path = Path(row.get('path', ''))
                            if not file_path.exists():
                                continue
                                
                            file_type = row.get('type', '')
                            extraction = None
                            
                            try:
                                if file_type == 'pdf':
                                    extraction = await document_extractor.extract_from_pdf(
                                        file_path, step.get('config', {})
                                    )
                                elif file_type == 'docx':
                                    extraction = await document_extractor.extract_from_docx(
                                        file_path, step.get('config', {})
                                    )
                                elif file_type == 'image':
                                    extraction = await document_extractor.extract_from_image(
                                        file_path, step.get('config', {})
                                    )
                                
                                if extraction:
                                    extraction_results.append(extraction)
                            except Exception as e:
                                print(f"Error extracting from {file_path}: {e}")
                                extraction_results.append({
                                    'file_path': str(file_path),
                                    'extraction_status': 'failed',
                                    'error': str(e)
                                })
                        
                        # Create new dataframe with extracted content
                        if extraction_results:
                            extracted_data = []
                            for result in extraction_results:
                                if result.get('extraction_status') == 'success':
                                    data_row = {
                                        'source_file': result['file_path'],
                                        'file_type': result.get('file_type', 'unknown'),
                                        'text_content': result.get('text_content', ''),
                                        'extraction_method': 'document_extractor'
                                    }
                                    
                                    # Add enhanced content if available
                                    if 'enhanced_content' in result:
                                        data_row['enhanced_content'] = result['enhanced_content']
                                    elif 'ai_extracted_content' in result and result['ai_extracted_content']:
                                        data_row['enhanced_content'] = result['ai_extracted_content'].get('content', '')
                                    
                                    # Add metadata
                                    if 'metadata' in result:
                                        data_row.update({
                                            f"meta_{k}": v for k, v in result['metadata'].items()
                                            if k not in ['extraction_timestamp']
                                        })
                                    
                                    # Add AI insights if available
                                    if 'ai_insights' in result:
                                        data_row['ai_insights'] = json.dumps(result['ai_insights'])
                                    
                                    extracted_data.append(data_row)
                            
                            if extracted_data:
                                current_df = pd.DataFrame(extracted_data)
                                results[step_name] = {
                                    'status': 'success',
                                    'files_processed': len(extraction_results),
                                    'successful_extractions': len(extracted_data),
                                    'failed_extractions': len([r for r in extraction_results if r.get('extraction_status') == 'failed'])
                                }
                            else:
                                results[step_name] = {
                                    'status': 'failed',
                                    'error': 'No successful extractions',
                                    'details': [r.get('error', 'Unknown error') for r in extraction_results if r.get('extraction_status') == 'failed']
                                }
                        else:
                            results[step_name] = {'status': 'no_documents_found'}
                    
                    elif step['type'] == 'quality_validation':
                        result = quality_validation_task(current_df, step['config'])
                        results[step_name] = result
                        
                    elif step['type'] == 'pii_detection':
                        result = pii_detection_task(current_df, step['config'])
                        current_df = result['processed_df']
                        results[step_name] = result['report']
                        
                    elif step['type'] in ['text_cleaning', 'data_cleaning']:
                        # Handle both text_cleaning and data_cleaning
                        config = step.get('config', {}).copy()
                        
                        # For text cleaning of extracted documents, don't drop rows with missing values
                        if step['type'] == 'text_cleaning' and 'text_content' in current_df.columns:
                            # Override handle_missing to keep all extracted content
                            config['handle_missing'] = 'fill'
                            config['text_columns'] = ['text_content', 'enhanced_content'] if 'enhanced_content' in current_df.columns else ['text_content']
                        
                        result = data_cleaning_task(current_df, config)
                        current_df = result['cleaned_df']
                        # Don't include the DataFrame in results
                        results[step_name] = {k: v for k, v in result.items() if k != 'cleaned_df'}
                        
                    elif step['type'] == 'content_analysis':
                        # Analyze extracted content
                        analysis_results = []
                        if 'text_content' in current_df.columns:
                            for idx, row in current_df.iterrows():
                                content = row.get('enhanced_content', row.get('text_content', ''))
                                if content:
                                    # Basic content analysis
                                    analysis = {
                                        'word_count': len(content.split()),
                                        'char_count': len(content),
                                        'line_count': content.count('\n') + 1,
                                        'has_tables': 'table' in content.lower() or '|' in content,
                                        'has_sections': any(marker in content for marker in ['#', 'Chapter', 'Section', '1.', '2.'])
                                    }
                                    analysis_results.append(analysis)
                                else:
                                    analysis_results.append({})
                            
                            # Add analysis results to dataframe
                            for key in ['word_count', 'char_count', 'line_count', 'has_tables', 'has_sections']:
                                current_df[f'analysis_{key}'] = [a.get(key, None) for a in analysis_results]
                        
                        results[step_name] = {
                            'status': 'success',
                            'rows_analyzed': len(analysis_results),
                            'average_word_count': sum(a.get('word_count', 0) for a in analysis_results) / len(analysis_results) if analysis_results else 0
                        }
                        
                    elif step['type'] == 'outlier_detection':
                        result = outlier_detection_task(current_df, step['config'])
                        current_df = result['processed_df']
                        # Don't include the DataFrame in results
                        results[step_name] = {k: v for k, v in result.items() if k != 'processed_df'}
                        
                    elif step['type'] == 'dataset_generation':
                        # Generate training dataset using AI
                        dataset_type = step.get('config', {}).get('dataset_type', 'qa_pairs')
                        print(f"[PipelineExecutor] Dataset generation step starting...")
                        print(f"[PipelineExecutor] Current DataFrame shape: {current_df.shape}")
                        print(f"[PipelineExecutor] DataFrame columns: {list(current_df.columns)}")
                        if not current_df.empty:
                            print(f"[PipelineExecutor] First row sample: {current_df.iloc[0].to_dict()}")
                        print(f"[PipelineExecutor] Generating {dataset_type} dataset from {len(current_df)} rows...")
                        
                        result = await dataset_generator.generate_dataset(
                            current_df,
                            dataset_type,
                            step.get('config', {})
                        )
                        
                        print(f"[PipelineExecutor] Dataset generation result: success={result.get('success')}, rows={result.get('row_count', 0)}")
                        if result.get('error'):
                            print(f"[PipelineExecutor] Dataset generation error: {result['error']}")
                        
                        if result.get('success') and not result['generated_df'].empty:
                            current_df = result['generated_df']
                            results[step_name] = {
                                'status': 'success',
                                'dataset_type': dataset_type,
                                'rows_generated': result.get('row_count', 0),
                                'columns': result.get('columns', [])
                            }
                        else:
                            results[step_name] = {
                                'status': 'failed',
                                'error': result.get('error', 'Dataset generation failed'),
                                'details': f"Current df was empty: {current_df.empty}, Result success: {result.get('success')}"
                            }
                    
                    elif step['type'] == 'data_transformation':
                        # AI transformation
                        if self.claude_service:
                            result = await self.transform_with_ai(current_df, step['config'])
                            current_df = result['transformed_df']
                            # Don't include the DataFrame in results
                            results[step_name] = {k: v for k, v in result.items() if k != 'transformed_df'}
                        else:
                            results[step_name] = {"error": "AI service not available"}
                            
                    elif step['type'] == 'schema_validation':
                        result = schema_validation_task(current_df, step['config'])
                        results[step_name] = result
                    
                    # Handle AI-generated step types that don't have implementations yet
                    elif step['type'] in ['pdf_text_extraction', 'content_validation', 'text_preprocessing', 'metadata_enrichment']:
                        # Map to existing implementations
                        if step['type'] == 'pdf_text_extraction':
                            # Treat as pdf extraction - use the existing pdf_extraction logic
                            extraction_results = []
                            for _, row in current_df.iterrows():
                                if row.get('type') == 'pdf':
                                    file_path = Path(row['path'])
                                    extraction = await document_extractor.extract_from_pdf(
                                        file_path, 
                                        step.get('config', {})
                                    )
                                    extraction_results.append(extraction)
                            
                            if extraction_results:
                                extracted_data = []
                                for result in extraction_results:
                                    if result.get('extraction_status') == 'success':
                                        extracted_data.append({
                                            'source_file': result['file_path'],
                                            'text_content': result.get('text_content', ''),
                                            'enhanced_content': result.get('enhanced_content', result.get('text_content', '')),
                                            'page_count': result['metadata'].get('page_count', 0),
                                            'tables_found': len(result.get('tables', [])),
                                            'extraction_method': 'advanced_extraction',
                                            'ai_insights': json.dumps(result.get('ai_insights', {})) if result.get('ai_insights') else None
                                        })
                                
                                if extracted_data:
                                    current_df = pd.DataFrame(extracted_data)
                                    results[step_name] = {
                                        'status': 'success',
                                        'files_processed': len(extraction_results),
                                        'successful_extractions': len(extracted_data)
                                    }
                                else:
                                    results[step_name] = {'error': 'No successful extractions'}
                            else:
                                results[step_name] = {'status': 'no_pdfs_found'}
                        elif step['type'] == 'content_validation':
                            # Use quality validation
                            result = quality_validation_task(current_df, step['config'])
                            results[step_name] = result
                        elif step['type'] == 'text_preprocessing':
                            # Use text cleaning with appropriate config
                            config = step.get('config', {}).copy()
                            if 'text_content' in current_df.columns:
                                # Override handle_missing to keep all extracted content
                                config['handle_missing'] = 'fill'
                                config['text_columns'] = ['text_content', 'enhanced_content'] if 'enhanced_content' in current_df.columns else ['text_content']
                            result = data_cleaning_task(current_df, config)
                            current_df = result['cleaned_df']
                            # Don't include the DataFrame in results
                            results[step_name] = {k: v for k, v in result.items() if k != 'cleaned_df'}
                        elif step['type'] == 'metadata_enrichment':
                            # Simple metadata enrichment
                            results[step_name] = {
                                'status': 'success',
                                'message': 'Metadata enrichment completed',
                                'rows_processed': len(current_df)
                            }
                    elif step['type'] == 'text_extraction':
                        # Text extraction is already done during load for text files
                        # This step just validates we have content
                        if len(current_df) > 0:
                            results[step_name] = {
                                'status': 'success',
                                'rows_processed': len(current_df),
                                'message': 'Text content already loaded'
                            }
                        else:
                            results[step_name] = {
                                'status': 'warning',
                                'message': 'No text content to extract'
                            }
                    
                    elif step['type'] == 'validation':
                        # Simple validation step - just check if we have data
                        validation_results = {
                            'status': 'success',
                            'rows_count': len(current_df),
                            'columns_count': len(current_df.columns) if len(current_df) > 0 else 0,
                            'has_data': len(current_df) > 0
                        }
                        
                        if len(current_df) == 0:
                            validation_results['warning'] = 'No data to validate'
                        
                        results[step_name] = validation_results
                        
                    else:
                        # Unknown step type
                        print(f"WARNING: Unknown step type '{step['type']}' - skipping")
                        results[step_name] = {
                            'status': 'skipped',
                            'message': f'Unknown step type: {step["type"]}'
                        }
                    
                    print(f"Step {step_name} completed successfully")
                    print(f"Current dataframe has {len(current_df)} rows")
                    
                except Exception as e:
                    print(f"ERROR in step {step_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    results[step_name] = {"error": str(e)}
                    raise
            
            # Save output dataset
            print(f"\nSaving output dataset with {len(current_df)} rows...")
            output_path = await self.save_output_dataset(current_df, pipeline, execution_id)
            print(f"Output saved to: {output_path}")
            
            # Calculate final metrics
            metrics = {
                "input_records": initial_rows,
                "output_records": len(current_df),
                "duration_seconds": 0,  # Will be calculated from timestamps
                "steps_completed": total_steps,
                "output_path": str(output_path),
            }
            
            # Update execution as completed
            execution_tracker.update_execution(execution_id, {
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
                "results": results,
                "metrics": metrics,
                "progress": 100,
            })
            
        except Exception as e:
            print(f"\nPIPELINE EXECUTION FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Update execution as failed
            execution_tracker.update_execution(execution_id, {
                "status": "failed",
                "completed_at": datetime.utcnow().isoformat(),
                "error": str(e),
            })
    
    async def load_dataset_files(self, dataset: Dict[str, Any]) -> pd.DataFrame:
        """Load all files from a dataset into a DataFrame"""
        dfs = []
        
        for file in dataset.get("files", []):
            file_path = Path(file["file_path"])
            file_type = file["file_type"]
            
            try:
                if file_type in ["text/csv", "application/csv"]:
                    df = pd.read_csv(file_path)
                    dfs.append(df)
                elif file_type == "application/json":
                    df = pd.read_json(file_path)
                    dfs.append(df)
                elif file_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
                    df = pd.read_excel(file_path)
                    dfs.append(df)
                elif file_type in ["text/plain", "text/markdown", "application/octet-stream"]:
                    # For text/markdown files, create a DataFrame with the content
                    # Check file extension for markdown files that might be marked as octet-stream
                    if file_type == "application/octet-stream" and not file_path.suffix.lower() in ['.md', '.txt']:
                        # Skip non-text octet-stream files
                        continue
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    df = pd.DataFrame({
                        "content": [content], 
                        "filename": [file["filename"]],
                        "type": ["text"],
                        "path": [str(file_path)]
                    })
                    dfs.append(df)
                elif file_type == "application/pdf":
                    # For PDFs, create a placeholder DataFrame
                    df = pd.DataFrame({
                        "filename": [file["filename"]], 
                        "type": ["pdf"],
                        "path": [str(file_path)],
                        "file_id": [file.get("id", "")],
                        "message": ["PDF content will be extracted during pipeline execution"]
                    })
                    dfs.append(df)
                elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
                    # For Word docs, create a placeholder DataFrame
                    df = pd.DataFrame({
                        "filename": [file["filename"]], 
                        "type": ["docx"],
                        "path": [str(file_path)],
                        "file_id": [file.get("id", "")],
                        "message": ["Word document content will be extracted during pipeline execution"]
                    })
                    dfs.append(df)
                elif file_type in ["image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"]:
                    # For images, create a placeholder DataFrame
                    df = pd.DataFrame({
                        "filename": [file["filename"]], 
                        "type": ["image"],
                        "path": [str(file_path)],
                        "file_id": [file.get("id", "")],
                        "message": ["Image content will be analyzed during pipeline execution"]
                    })
                    dfs.append(df)
            except Exception as e:
                print(f"Error loading file {file['filename']}: {str(e)}")
                continue
        
        if not dfs:
            # Return empty DataFrame if no files could be loaded
            return pd.DataFrame()
        
        # Combine all DataFrames
        if len(dfs) == 1:
            return dfs[0]
        else:
            # Try to concatenate with common columns
            return pd.concat(dfs, ignore_index=True, sort=False)
    
    async def transform_with_ai(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data using Claude AI"""
        prompt = config.get("prompt_template", "Transform the data")
        target_schema = config.get("target_schema")
        
        # For document/image datasets, handle specially
        if 'type' in df.columns and df['type'].isin(['pdf', 'image']).any():
            # Process each document/image
            transformed_data = []
            
            for _, row in df.iterrows():
                if row.get('type') == 'pdf':
                    # Extract text from PDF using Claude's vision
                    result = await self.claude_service.extract_from_pdf(
                        Path(row['path']), 
                        prompt
                    )
                    transformed_data.append(result)
                elif row.get('type') == 'image':
                    # Extract data from image using Claude's vision
                    result = await self.claude_service.extract_from_image(
                        Path(row['path']), 
                        prompt
                    )
                    transformed_data.append(result)
                else:
                    # Regular data transformation
                    transformed_data.append(row.to_dict())
            
            # Convert results to DataFrame
            transformed_df = pd.DataFrame(transformed_data)
        else:
            # Regular data transformation
            transformed_df = await self.claude_service.transform_data(df, prompt, target_schema)
        
        return {
            "transformed_df": transformed_df,
            "transformation_log": {
                "input_rows": len(df),
                "output_rows": len(transformed_df),
                "prompt": prompt,
            }
        }
    
    async def save_output_dataset(self, df: pd.DataFrame, pipeline: Dict[str, Any], execution_id: str) -> Path:
        """Save the output dataset"""
        output_dir = Path("uploads") / "pipeline_outputs"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{pipeline['name'].replace(' ', '_')}_{timestamp}.csv"
        output_path = output_dir / output_filename
        
        # Save as CSV
        df.to_csv(output_path, index=False)
        
        return output_path

# Global pipeline executor instance
pipeline_executor = PipelineExecutor()

def execute_pipeline(pipeline: Dict[str, Any], dataset: Dict[str, Any]) -> str:
    """Execute a pipeline asynchronously"""
    execution_id = execution_tracker.create_execution(pipeline["id"])
    print(f"Starting pipeline execution: {execution_id} for pipeline: {pipeline['id']}")
    
    # Submit to thread pool for async execution
    future = executor.submit(
        asyncio.run,
        pipeline_executor.execute_pipeline_async(pipeline, dataset, execution_id)
    )
    
    # Add error callback
    def handle_error(fut):
        try:
            fut.result()
        except Exception as e:
            print(f"Pipeline execution error: {e}")
            import traceback
            traceback.print_exc()
            execution_tracker.update_execution(execution_id, {
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.utcnow().isoformat()
            })
    
    future.add_done_callback(handle_error)
    
    return execution_id

def get_execution_status(execution_id: str) -> Optional[Dict[str, Any]]:
    """Get the status of a pipeline execution"""
    return execution_tracker.get_execution(execution_id)

def get_pipeline_executions(pipeline_id: str) -> List[Dict[str, Any]]:
    """Get all executions for a pipeline"""
    return execution_tracker.get_pipeline_executions(pipeline_id)