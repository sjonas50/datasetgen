"""
AI-Powered Pipeline Analyzer
Analyzes uploaded documents and automatically suggests/creates appropriate pipelines
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime

from services.claude_service import ClaudeService
import os
from anthropic import AsyncAnthropic

class PipelineAnalyzer:
    def __init__(self):
        self.claude_service = ClaudeService()
        
    async def analyze_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a dataset and suggest appropriate pipeline steps
        """
        # Extract sample data from files
        sample_data = await self._extract_sample_data(dataset)
        
        # Analyze with AI
        analysis = await self._analyze_with_ai(sample_data, dataset)
        
        # Generate pipeline configuration
        pipeline_config = self._generate_pipeline_config(analysis)
        
        return {
            "analysis": analysis,
            "suggested_pipeline": pipeline_config,
            "confidence_score": analysis.get("confidence", 0.9)
        }
    
    async def _extract_sample_data(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Extract sample data from dataset files"""
        samples = []
        
        for file in dataset.get("files", [])[:5]:  # Analyze up to 5 files
            file_path = Path(file["file_path"])
            file_type = file["file_type"]
            
            sample = {
                "filename": file["filename"],
                "file_type": file_type,
                "size": file["size"]
            }
            
            try:
                if file_type in ["text/csv", "application/csv"]:
                    df = pd.read_csv(file_path, nrows=10)
                    sample["data_preview"] = {
                        "columns": list(df.columns),
                        "sample_rows": df.head(5).to_dict(orient="records"),
                        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
                    }
                elif file_type == "application/json":
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        sample["data_preview"] = data[:5]
                    else:
                        sample["data_preview"] = data
                elif file_type in ["text/plain", "text/markdown"]:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read(1000)  # First 1000 chars
                    sample["text_preview"] = content
                elif file_type == "application/pdf":
                    sample["document_type"] = "pdf"
                    sample["needs_extraction"] = True
                elif file_type in ["image/png", "image/jpeg", "image/jpg"]:
                    sample["document_type"] = "image"
                    sample["needs_vision_processing"] = True
            except Exception as e:
                sample["error"] = str(e)
            
            samples.append(sample)
        
        return {
            "dataset_name": dataset["name"],
            "dataset_description": dataset.get("description", ""),
            "file_samples": samples,
            "total_files": len(dataset.get("files", []))
        }
    
    async def _analyze_with_ai(self, sample_data: Dict[str, Any], dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Use Claude to analyze the data and suggest pipeline steps"""
        
        # Ensure we have the API key
        if not self.claude_service.enabled:
            print("Claude service not enabled, using default analysis")
            return self._get_default_analysis(sample_data)
        
        prompt = f"""
        Analyze this dataset and suggest appropriate data processing pipeline steps.
        
        Dataset Name: {sample_data['dataset_name']}
        Description: {sample_data['dataset_description']}
        Total Files: {sample_data['total_files']}
        
        File Samples:
        {json.dumps(sample_data['file_samples'], indent=2)}
        
        Based on this data, please:
        1. Identify the type of data (structured, unstructured, documents, etc.)
        2. Detect potential data quality issues
        3. Identify if there's sensitive/PII data
        4. Suggest appropriate processing steps
        5. Include a dataset_generation step to create training data
        6. Recommend the output format
        
        Respond in JSON format with:
        {{
            "data_type": "structured|unstructured|documents|mixed",
            "detected_issues": ["list of potential issues"],
            "contains_pii": true/false,
            "pii_types": ["if PII detected, what types"],
            "recommended_steps": [
                {{
                    "type": "step_type",
                    "reason": "why this step is needed",
                    "config": {{}}
                }}
            ],
            "output_format": "csv|json|parquet",
            "insights": "Brief analysis of the data",
            "confidence": 0.0-1.0
        }}
        """
        
        try:
            # Use Claude Sonnet 4 for analysis
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Call Claude API using async client
            response = await self.claude_service.client.messages.create(
                model="claude-sonnet-4-20250514",  # Claude Sonnet 4
                max_tokens=2000,
                temperature=0.2,  # Lower temperature for more consistent JSON output
                messages=messages
            )
            
            # Extract the response text
            response_text = response.content[0].text
            
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                print(f"Could not parse JSON from Claude response: {response_text[:200]}...")
                return self._get_default_analysis(sample_data)
                
        except Exception as e:
            print(f"AI analysis error: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_analysis(sample_data)
    
    def _get_default_analysis(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide default analysis when AI is not available"""
        # Detect data type based on file types
        file_types = [f["file_type"] for f in sample_data["file_samples"]]
        
        if any("csv" in ft or "excel" in ft for ft in file_types):
            data_type = "structured"
        elif any("pdf" in ft or "doc" in ft for ft in file_types):
            data_type = "documents"
        else:
            data_type = "mixed"
        
        # Default pipeline steps based on data type
        if data_type == "structured":
            steps = [
                {"type": "quality_validation", "reason": "Ensure data quality"},
                {"type": "data_cleaning", "reason": "Handle missing values and duplicates"},
                {"type": "pii_detection", "reason": "Check for sensitive data"}
            ]
        elif data_type == "documents":
            steps = [
                {"type": "document_extraction", "reason": "Extract content from documents using AI"},
                {"type": "text_cleaning", "reason": "Clean and normalize extracted text"},
                {"type": "content_analysis", "reason": "Analyze content structure and metadata"},
                {"type": "pii_detection", "reason": "Check for sensitive information"}
            ]
        else:
            steps = [
                {"type": "quality_validation", "reason": "Validate data quality"},
                {"type": "pii_detection", "reason": "Detect sensitive information"}
            ]
        
        return {
            "data_type": data_type,
            "detected_issues": ["Unable to perform deep analysis - AI service unavailable"],
            "contains_pii": True,  # Assume true for safety
            "pii_types": ["unknown"],
            "recommended_steps": steps,
            "output_format": "csv",
            "insights": "Basic analysis performed. Enable AI service for deeper insights.",
            "confidence": 0.5
        }
    
    def _generate_pipeline_config(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate pipeline configuration based on analysis"""
        steps = []
        
        for idx, rec_step in enumerate(analysis["recommended_steps"]):
            step_type = rec_step["type"]
            
            # Map analysis recommendations to actual pipeline steps
            if step_type == "quality_validation":
                step = {
                    "id": f"step_{idx}",
                    "type": "quality_validation",
                    "name": "Data Quality Check",
                    "config": {
                        "check_missing": True,
                        "check_duplicates": True,
                        "check_types": True
                    }
                }
            elif step_type == "pii_detection":
                step = {
                    "id": f"step_{idx}",
                    "type": "pii_detection",
                    "name": "PII Detection",
                    "config": {
                        "scan_mode": "comprehensive",
                        "pii_types": analysis.get("pii_types", ["all"])
                    }
                }
            elif step_type == "data_cleaning":
                step = {
                    "id": f"step_{idx}",
                    "type": "data_cleaning",
                    "name": "Data Cleaning",
                    "config": {
                        "handle_missing": "smart_impute",
                        "remove_duplicates": True,
                        "standardize_formats": True
                    }
                }
            elif step_type == "document_extraction":
                step = {
                    "id": f"step_{idx}",
                    "type": "document_extraction",
                    "name": "Extract content from documents",
                    "config": {
                        "extraction_method": "ai_enhanced",
                        "use_ai_extraction": True,
                        "extract_images": True,
                        "extraction_prompt": "Extract all text content, tables, and important information from this document. Include metadata and structure."
                    }
                }
            elif step_type == "text_cleaning":
                step = {
                    "id": f"step_{idx}",
                    "type": "text_cleaning",
                    "name": "Clean and normalize extracted text",
                    "config": {
                        "remove_special_chars": True,
                        "normalize_whitespace": True,
                        "remove_headers_footers": True,
                        "preserve_structure": True
                    }
                }
            elif step_type == "content_analysis":
                step = {
                    "id": f"step_{idx}",
                    "type": "content_analysis",
                    "name": "Analyze content for structure and metadata",
                    "config": {
                        "detect_sections": True,
                        "extract_metadata": True,
                        "identify_tables": True,
                        "calculate_statistics": True
                    }
                }
            elif step_type == "data_transformation":
                step = {
                    "id": f"step_{idx}",
                    "type": "data_transformation",
                    "name": "AI Data Transformation",
                    "config": {
                        "prompt_template": "Transform this data into a structured format",
                        "output_format": analysis.get("output_format", "csv")
                    }
                }
            elif step_type == "dataset_generation":
                step = {
                    "id": f"step_{idx}",
                    "type": "dataset_generation",
                    "name": "Generate Training Dataset",
                    "config": {
                        "dataset_type": "qa_pairs",  # Default to Q&A pairs
                        "min_examples": 20,
                        "additional_instructions": "Generate high-quality training data"
                    }
                }
            else:
                # Generic step
                step = {
                    "id": f"step_{idx}",
                    "type": step_type,
                    "name": rec_step.get("reason", step_type),
                    "config": rec_step.get("config", {})
                }
            
            steps.append(step)
        
        return {
            "name": f"Auto-Generated Pipeline for {analysis.get('data_type', 'mixed')} data",
            "description": f"AI-generated pipeline based on data analysis. {analysis.get('insights', '')}",
            "steps": steps,
            "auto_generated": True,
            "confidence": analysis.get("confidence", 0.5)
        }

# Singleton instance
pipeline_analyzer = PipelineAnalyzer()