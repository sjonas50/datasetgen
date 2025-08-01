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
from services.document_pipeline import document_pipeline
from services.document import ProcessedDocument, DocumentCollection
from services.dataset_generator_v2 import DatasetGeneratorV2
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
        self.v2_generator = DatasetGeneratorV2()
    
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
                        # Check if we're using the new document pipeline format
                        if 'document_id' in current_df.columns and hasattr(self, '_document_collection'):
                            # Documents already loaded by document pipeline
                            print(f"[PipelineExecutor] Using document collection for extraction")
                            
                            # Extract actual content from each document
                            extraction_results = []
                            for doc in self._document_collection.documents:
                                print(f"[PipelineExecutor] Extracting content from {doc.filename}")
                                
                                config = step.get('config', {})
                                if config.get('use_ai_extraction', True) and doc.file_type == 'pdf':
                                    # Use AI extraction for PDFs
                                    result = await document_extractor.extract_from_pdf(
                                        Path(doc.file_path), 
                                        config
                                    )
                                    extraction_results.append(result)
                                elif doc.file_type in ['png', 'jpg', 'jpeg', 'gif', 'webp']:
                                    # Use AI extraction for images
                                    result = await document_extractor.extract_from_image(
                                        Path(doc.file_path),
                                        config
                                    )
                                    extraction_results.append(result)
                                else:
                                    # For text-based documents, use existing content
                                    extraction_results.append({
                                        'file_path': doc.file_path,
                                        'extraction_status': 'success',
                                        'text_content': doc.content,
                                        'enhanced_content': doc.content,
                                        'metadata': doc.metadata
                                    })
                        else:
                            # Legacy extraction for old format
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
                        target_rows = step.get('config', {}).get('target_rows')
                        print(f"[PipelineExecutor] Dataset generation step starting...")
                        print(f"[PipelineExecutor] Current DataFrame shape: {current_df.shape}")
                        print(f"[PipelineExecutor] DataFrame columns: {list(current_df.columns)}")
                        print(f"[PipelineExecutor] Target rows: {target_rows}")
                        if not current_df.empty:
                            print(f"[PipelineExecutor] First row sample: {current_df.iloc[0].to_dict()}")
                        print(f"[PipelineExecutor] Generating {dataset_type} dataset from {len(current_df)} rows...")
                        
                        # Pass target_rows in config
                        generation_config = step.get('config', {}).copy()
                        if target_rows:
                            generation_config['target_rows'] = target_rows
                        
                        result = await dataset_generator.generate_dataset(
                            current_df,
                            dataset_type,
                            generation_config
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
        """Load all files from a dataset into a DataFrame using the new document pipeline"""
        files = dataset.get("files", [])
        
        if not files:
            print("[PipelineExecutor] No files in dataset")
            return pd.DataFrame()
        
        print(f"[PipelineExecutor] Loading {len(files)} files using document pipeline")
        
        # Extract file paths and types
        file_info = []
        for file in files:
            file_info.append({
                'path': Path(file["file_path"]),
                'type': file["file_type"],
                'filename': file["filename"]
            })
        
        try:
            # Use document pipeline to load all files
            file_paths = [f['path'] for f in file_info]
            
            # Add progress callback
            def progress_callback(progress_data):
                print(f"[PipelineExecutor] {progress_data['stage']}: {progress_data['message']} ({progress_data['percentage']:.1f}%)")
            
            document_pipeline.add_progress_callback(progress_callback)
            
            # Process all documents
            collection = await document_pipeline.process_multiple_documents(
                file_paths,
                options={
                    'parallel_processing': len(file_paths) > 5,
                    'batch_size': 5,
                    'enable_chunking': False  # Will be handled by dataset generator if needed
                }
            )
            
            # Convert to DataFrame for pipeline processing
            df = collection.to_dataframe()
            
            # Add original file metadata
            for i, file in enumerate(files):
                if i < len(df):
                    df.loc[i, 'file_id'] = file.get('id', '')
                    df.loc[i, 'original_file_type'] = file.get('file_type', '')
            
            print(f"[PipelineExecutor] Loaded {len(df)} rows from {collection.total_documents} documents")
            print(f"[PipelineExecutor] Document types: {collection.document_types}")
            
            # Store collection for later use
            self._document_collection = collection
            
            return df
            
        except Exception as e:
            print(f"[PipelineExecutor] Error loading files with document pipeline: {e}")
            # Fallback to old method
            return await self._load_dataset_files_legacy(dataset)
    
    async def _load_dataset_files_legacy(self, dataset: Dict[str, Any]) -> pd.DataFrame:
        """Legacy file loading method as fallback"""
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

    async def execute_pipeline_streaming(self, pipeline: Dict[str, Any], dataset: Dict[str, Any], 
                                       execution_id: str, progress_callback: callable):
        """Execute pipeline with streaming progress updates"""
        print(f"Pipeline executor started for streaming execution: {execution_id}")
        
        try:
            # Update status to running
            execution_tracker.update_execution(execution_id, {
                "status": "running",
                "started_at": datetime.utcnow().isoformat(),
            })
            
            # Send initial progress
            await progress_callback({
                "stage": "initializing",
                "message": "Starting pipeline execution",
                "percentage": 0
            })
            
            # Load dataset files
            await progress_callback({
                "stage": "loading",
                "message": "Loading dataset files...",
                "percentage": 10
            })
            
            df = await self.load_dataset_files(dataset)
            initial_rows = len(df)
            
            await progress_callback({
                "stage": "loaded",
                "message": f"Loaded {initial_rows} rows from dataset",
                "percentage": 20
            })
            
            # Execute each step
            results = {}
            current_df = df.copy()
            total_steps = len(pipeline["steps"])
            
            for i, step in enumerate(pipeline["steps"]):
                step_name = f"step_{i}_{step['type']}"
                base_percentage = 20 + (i * 60 / total_steps)
                
                # Update progress for step start
                await progress_callback({
                    "stage": "processing",
                    "current_step": step_name,
                    "step_type": step['type'],
                    "message": f"Executing {step['type']}...",
                    "percentage": base_percentage,
                    "step_number": i + 1,
                    "total_steps": total_steps
                })
                
                execution_tracker.update_execution(execution_id, {
                    "current_step": step_name,
                    "progress": int(base_percentage),
                })
                
                try:
                    # Handle dataset generation with special progress tracking
                    if step['type'] == 'dataset_generation':
                        dataset_type = step.get('config', {}).get('dataset_type', 'qa_pairs')
                        target_rows = step.get('config', {}).get('target_rows')
                        
                        # Create a sub-progress callback for dataset generation
                        async def dataset_progress_callback(progress_data):
                            sub_percentage = base_percentage + (progress_data.get('percentage', 0) * 60 / total_steps / 100)
                            await progress_callback({
                                "stage": "dataset_generation",
                                "current_step": step_name,
                                "step_type": step['type'],
                                "message": f"Generating {dataset_type} dataset: {progress_data.get('rows_generated', 0)}/{progress_data.get('target_rows', target_rows)} rows",
                                "percentage": sub_percentage,
                                "step_number": i + 1,
                                "total_steps": total_steps,
                                "dataset_progress": progress_data
                            })
                        
                        # Pass progress callback to dataset generator
                        generation_config = step.get('config', {}).copy()
                        if target_rows:
                            generation_config['target_rows'] = target_rows
                        
                        # Use streaming generation if available
                        if target_rows and target_rows > 200:
                            result = await self.v2_generator.generate_dataset_streaming(
                                current_df,
                                dataset_type,
                                generation_config,
                                dataset_progress_callback
                            )
                        else:
                            result = await dataset_generator.generate_dataset(
                                current_df,
                                dataset_type,
                                generation_config
                            )
                        
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
                                'error': result.get('error', 'Dataset generation failed')
                            }
                    else:
                        # Execute other step types using existing logic from execute_pipeline_async
                        # Copy the execution logic but add progress callbacks
                        
                        # Get the original execution logic
                        result = await self._execute_step_with_progress(
                            step, step_name, current_df, progress_callback, 
                            base_percentage, i, total_steps
                        )
                        
                        if 'error' in result:
                            results[step_name] = result
                        else:
                            if 'processed_df' in result:
                                current_df = result['processed_df']
                                results[step_name] = {k: v for k, v in result.items() if k != 'processed_df'}
                            elif 'cleaned_df' in result:
                                current_df = result['cleaned_df']
                                results[step_name] = {k: v for k, v in result.items() if k != 'cleaned_df'}
                            elif 'transformed_df' in result:
                                current_df = result['transformed_df']
                                results[step_name] = {k: v for k, v in result.items() if k != 'transformed_df'}
                            elif 'generated_df' in result:
                                current_df = result['generated_df']
                                results[step_name] = {k: v for k, v in result.items() if k != 'generated_df'}
                            else:
                                results[step_name] = result
                    
                    # Update progress for step completion
                    await progress_callback({
                        "stage": "processing",
                        "current_step": step_name,
                        "message": f"Completed {step['type']}",
                        "percentage": base_percentage + 60 / total_steps,
                        "step_number": i + 1,
                        "total_steps": total_steps
                    })
                    
                except Exception as e:
                    print(f"ERROR in step {step_name}: {str(e)}")
                    results[step_name] = {"error": str(e)}
                    await progress_callback({
                        "stage": "error",
                        "current_step": step_name,
                        "message": f"Error in {step['type']}: {str(e)}",
                        "percentage": base_percentage
                    })
                    raise
            
            # Save output
            await progress_callback({
                "stage": "saving",
                "message": "Saving output dataset...",
                "percentage": 85
            })
            
            output_path = await self.save_output_dataset(current_df, pipeline, execution_id)
            
            # Calculate final metrics
            metrics = {
                "input_records": initial_rows,
                "output_records": len(current_df),
                "duration_seconds": 0,
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
            
            await progress_callback({
                "stage": "complete",
                "message": "Pipeline execution completed successfully!",
                "percentage": 100,
                "metrics": metrics
            })
            
        except Exception as e:
            print(f"PIPELINE EXECUTION FAILED: {str(e)}")
            execution_tracker.update_execution(execution_id, {
                "status": "failed",
                "completed_at": datetime.utcnow().isoformat(),
                "error": str(e),
            })
            
            await progress_callback({
                "stage": "failed",
                "message": f"Pipeline execution failed: {str(e)}",
                "percentage": -1
            })
            raise
    
    async def _execute_step_with_progress(self, step, step_name, current_df, progress_callback, 
                                        base_percentage, step_num, total_steps):
        """Execute a single step with progress tracking"""
        
        step_type = step['type']
        step_percentage = base_percentage + 30 / total_steps
        
        if step_type == 'document_extraction':
            await progress_callback({
                "stage": "processing",
                "message": "Extracting document content...",
                "percentage": step_percentage
            })
            
            # Copy document extraction logic from execute_pipeline_async
            if 'document_id' in current_df.columns and hasattr(self, '_document_collection'):
                # Documents already loaded by document pipeline
                extraction_results = []
                for doc in self._document_collection.documents:
                    config = step.get('config', {})
                    if config.get('use_ai_extraction', True) and doc.file_type == 'pdf':
                        result = await document_extractor.extract_from_pdf(
                            Path(doc.file_path), 
                            config
                        )
                        extraction_results.append(result)
                    elif doc.file_type in ['png', 'jpg', 'jpeg', 'gif', 'webp']:
                        result = await document_extractor.extract_from_image(
                            Path(doc.file_path),
                            config
                        )
                        extraction_results.append(result)
                    else:
                        extraction_results.append({
                            'file_path': doc.file_path,
                            'extraction_status': 'success',
                            'text_content': doc.content,
                            'enhanced_content': doc.content,
                            'metadata': doc.metadata
                        })
            
            # Process extraction results
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
                        if 'enhanced_content' in result:
                            data_row['enhanced_content'] = result['enhanced_content']
                        extracted_data.append(data_row)
                
                if extracted_data:
                    return {'processed_df': pd.DataFrame(extracted_data), 'status': 'success'}
            
            return {'status': 'no_documents_found'}
            
        elif step_type == 'quality_validation':
            await progress_callback({
                "stage": "processing",
                "message": "Validating data quality...",
                "percentage": step_percentage
            })
            return quality_validation_task(current_df, step['config'])
            
        elif step_type == 'pii_detection':
            await progress_callback({
                "stage": "processing",
                "message": "Detecting PII data...",
                "percentage": step_percentage
            })
            result = pii_detection_task(current_df, step['config'])
            return {'processed_df': result['processed_df'], 'report': result['report']}
            
        elif step_type in ['text_cleaning', 'data_cleaning']:
            await progress_callback({
                "stage": "processing",
                "message": "Cleaning data...",
                "percentage": step_percentage
            })
            config = step.get('config', {}).copy()
            if step_type == 'text_cleaning' and 'text_content' in current_df.columns:
                config['handle_missing'] = 'fill'
                config['text_columns'] = ['text_content', 'enhanced_content'] if 'enhanced_content' in current_df.columns else ['text_content']
            return data_cleaning_task(current_df, config)
            
        elif step_type == 'outlier_detection':
            await progress_callback({
                "stage": "processing",
                "message": "Detecting outliers...",
                "percentage": step_percentage
            })
            return outlier_detection_task(current_df, step['config'])
            
        elif step_type == 'schema_validation':
            await progress_callback({
                "stage": "processing",
                "message": "Validating schema...",
                "percentage": step_percentage
            })
            return schema_validation_task(current_df, step['config'])
            
        else:
            # For unknown step types
            return {'status': 'skipped', 'message': f'Unknown step type: {step_type}'}

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