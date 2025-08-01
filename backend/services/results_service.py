"""
Pipeline Results Service
Manages and retrieves pipeline execution results
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime

class ResultsService:
    def __init__(self):
        self.output_dir = Path("uploads/pipeline_outputs")
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def get_execution_output_path(self, execution_id: str) -> Optional[Path]:
        """Find the output file for a specific execution"""
        # Look for files that might contain the execution ID or timestamp
        for file in self.output_dir.iterdir():
            if file.is_file() and file.suffix == '.csv':
                # Check if file was created around the same time as execution
                # This is a simple implementation - in production, store the path with execution
                return file
        return None
    
    def get_execution_results(self, execution_id: str) -> Dict[str, Any]:
        """Get detailed results for an execution"""
        from services.pipeline_executor import get_execution_status
        
        # Get execution status first
        execution = get_execution_status(execution_id)
        if not execution:
            return {
                "status": "not_found",
                "message": "Execution not found"
            }
        
        # Find output file
        output_path = None
        if execution.get("metrics", {}).get("output_path"):
            output_path = Path(execution["metrics"]["output_path"])
        else:
            output_path = self.get_execution_output_path(execution_id)
        
        if not output_path or not output_path.exists():
            return {
                "status": "no_output",
                "message": "No output file found for this execution",
                "execution_status": execution["status"],
                "step_results": execution.get("results", {})
            }
        
        try:
            # Read the output file
            df = pd.read_csv(output_path)
            
            result = {
                "status": "success",
                "output_file": str(output_path),
                "file_size": output_path.stat().st_size,
                "rows": len(df),
                "columns": list(df.columns),
                "execution_status": execution["status"],
                "step_results": execution.get("results", {}),
                "metrics": execution.get("metrics", {}),
                "statistics": {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "file_size_kb": round(output_path.stat().st_size / 1024, 2)
                }
            }
            
            # Check if this is document extraction output
            if 'text_content' in df.columns or 'enhanced_content' in df.columns:
                # Provide detailed document extraction preview
                extracted_documents = []
                for idx, row in df.head(10).iterrows():
                    doc_info = {
                        "index": idx,
                        "source_file": row.get("source_file", "Unknown"),
                        "file_type": row.get("file_type", "Unknown")
                    }
                    
                    # Get content preview
                    content = row.get("enhanced_content") or row.get("text_content", "")
                    if content:
                        doc_info["content_preview"] = content[:500] + "..." if len(content) > 500 else content
                        doc_info["content_length"] = len(content)
                        doc_info["word_count"] = len(content.split())
                    
                    # Get metadata
                    metadata = {}
                    for col in df.columns:
                        if col.startswith("meta_") and pd.notna(row[col]):
                            metadata[col.replace("meta_", "")] = row[col]
                    if metadata:
                        doc_info["metadata"] = metadata
                    
                    # Get AI insights if available
                    if "ai_insights" in row and pd.notna(row["ai_insights"]):
                        try:
                            doc_info["ai_insights"] = json.loads(row["ai_insights"])
                        except:
                            doc_info["ai_insights"] = row["ai_insights"]
                    
                    extracted_documents.append(doc_info)
                
                result["document_extraction"] = {
                    "total_documents": len(df),
                    "preview_count": len(extracted_documents),
                    "documents": extracted_documents,
                    "has_enhanced_content": 'enhanced_content' in df.columns,
                    "has_ai_insights": 'ai_insights' in df.columns
                }
            else:
                # Standard data preview
                result["preview"] = df.head(10).to_dict(orient='records')
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error reading output file: {str(e)}",
                "execution_status": execution["status"]
            }
    
    def download_results(self, execution_id: str) -> Optional[tuple]:
        """Get file path and name for downloading results"""
        output_path = self.get_execution_output_path(execution_id)
        
        if output_path and output_path.exists():
            return str(output_path), output_path.name
        
        return None

# Singleton instance
results_service = ResultsService()