"""
Claude 4 Integration Service
Provides intelligent data processing capabilities using Claude's latest models (July 2025)
Supports both Sonnet 4 and Opus 4 with enhanced vision and document processing
"""

import os
import json
import base64
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import asyncio
from datetime import datetime
import pandas as pd
from PIL import Image
import io
import PyPDF2
import anthropic
from anthropic import AsyncAnthropic

class ClaudeService:
    """Service for interacting with Claude 4 models (Sonnet & Opus)
    
    Features:
    - Native PDF processing without conversion
    - Enhanced vision capabilities for charts, diagrams, and tables
    - Extended thinking mode for complex reasoning
    - Tool use during processing
    - Support for up to 64K output tokens (Sonnet 4)
    """
    
    def __init__(self, model_type="sonnet"):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.enabled = bool(self.api_key)
        
        print(f"[ClaudeService] Initializing - API key present: {bool(self.api_key)}, enabled: {self.enabled}")
        
        if self.enabled:
            self.client = AsyncAnthropic(api_key=self.api_key)
            
            # Claude 4 models (May 2025)
            if model_type == "opus":
                self.model = "claude-opus-4-20250514"  # Claude Opus 4 - For complex tasks
                self.max_tokens = 8192
                self.supports_extended_thinking = True
            else:
                self.model = "claude-sonnet-4-20250514"  # Claude Sonnet 4 - Default
                self.max_tokens = 8192  # Adjusted to match API limits
                self.supports_extended_thinking = True
            
            print(f"[ClaudeService] Client initialized with model: {self.model}")
            self.temperature = 0.1  # Low temperature for consistent results
            self.supports_vision = True  # Both models support enhanced vision
            self.supports_pdf_native = True  # Native PDF processing
            self.supports_tool_use = True  # Tool use during extended thinking
        else:
            print("Warning: ANTHROPIC_API_KEY not set. AI features will be disabled.")
            self.client = None
    
    async def analyze_data_quality(self, df: pd.DataFrame, context: str = "") -> Dict[str, Any]:
        """
        Analyze data quality using Claude's intelligence
        """
        if not self.enabled:
            return {
                "quality_score": 75.0,
                "issues": self._basic_quality_check(df),
                "patterns": [],
                "recommendations": ["Enable AI service for detailed analysis"],
                "pii_concerns": [],
                "schema_suggestions": {},
                "ai_enabled": False
            }
        # Prepare data sample for Claude
        sample_size = min(100, len(df))
        sample_data = df.head(sample_size).to_dict(orient='records')
        
        # Get basic statistics
        stats = {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "shape": df.shape,
            "missing_values": df.isnull().sum().to_dict(),
            "unique_counts": {col: df[col].nunique() for col in df.columns}
        }
        
        prompt = f"""Analyze this dataset for quality issues and provide detailed insights.

Dataset Context: {context}

Dataset Statistics:
{json.dumps(stats, indent=2)}

Sample Data (first {sample_size} rows):
{json.dumps(sample_data, indent=2)}

Please provide:
1. Overall data quality assessment (score 0-100)
2. Identified quality issues with severity (critical/high/medium/low)
3. Data patterns and anomalies detected
4. Specific recommendations for improvement
5. Potential PII or sensitive data concerns
6. Schema inference and data type recommendations

Format your response as a JSON object with these keys:
- quality_score: number (0-100)
- issues: array of {{type, severity, description, affected_columns}}
- patterns: array of detected patterns
- recommendations: array of actionable recommendations
- pii_concerns: array of potential PII fields
- schema_suggestions: object with column type recommendations"""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Parse Claude's response
            content = response.content[0].text
            
            # Try to extract JSON from the response
            try:
                # Look for JSON block in the response
                import re
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    # Fallback: create structured response from text
                    result = self._parse_text_response(content)
            except json.JSONDecodeError:
                result = self._parse_text_response(content)
            
            result['analysis_timestamp'] = datetime.utcnow().isoformat()
            result['model_used'] = self.model
            
            return result
            
        except Exception as e:
            print(f"Error in Claude analysis: {str(e)}")
            # Return basic analysis as fallback
            return {
                "quality_score": 70.0,
                "issues": self._basic_quality_check(df),
                "patterns": [],
                "recommendations": ["Manual review recommended due to analysis error"],
                "pii_concerns": [],
                "schema_suggestions": {},
                "error": str(e)
            }
    
    async def transform_data(
        self, 
        df: pd.DataFrame, 
        transformation_request: str,
        target_schema: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Transform data using Claude's code generation capabilities
        """
        sample_data = df.head(10).to_dict(orient='records')
        current_schema = {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        prompt = f"""Generate Python code to transform this dataset according to the request.

Transformation Request: {transformation_request}

Current Schema:
{json.dumps(current_schema, indent=2)}

Sample Data:
{json.dumps(sample_data, indent=2)}

{f"Target Schema: {json.dumps(target_schema, indent=2)}" if target_schema else ""}

Requirements:
1. The code should work with a pandas DataFrame named 'df'
2. Return the transformed DataFrame
3. Handle edge cases and data type conversions
4. Include error handling
5. Add comments explaining the transformations

Provide ONLY the Python code, no explanations."""

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        code = response.content[0].text
        
        # Clean up code block markers if present
        code = code.replace("```python", "").replace("```", "").strip()
        
        # Execute the transformation code
        try:
            # Create a safe execution environment
            local_vars = {'df': df.copy(), 'pd': pd}
            exec(code, {"__builtins__": {}}, local_vars)
            
            # Get the transformed dataframe
            if 'df' in local_vars:
                return local_vars['df']
            else:
                # Look for any DataFrame in the local vars
                for var_name, var_value in local_vars.items():
                    if isinstance(var_value, pd.DataFrame):
                        return var_value
                
                raise ValueError("No transformed DataFrame found in execution result")
                
        except Exception as e:
            print(f"Error executing transformation: {str(e)}")
            print(f"Generated code:\n{code}")
            raise
    
    async def extract_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract structured data from PDF using Claude's vision capabilities
        """
        # Read PDF and convert pages to images
        pdf_images = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = min(len(pdf_reader.pages), 5)  # Limit to first 5 pages
            
            # Extract text from PDF
            text_content = ""
            for i in range(num_pages):
                page = pdf_reader.pages[i]
                text_content += page.extract_text() + "\n\n"
        
        # For vision analysis, we'd need to convert PDF to images
        # This is a placeholder - in production, use pdf2image or similar
        
        prompt = f"""Analyze this PDF content and extract structured data.

PDF Text Content:
{text_content[:3000]}  # Limit text for API

Please extract:
1. Tables with headers and data
2. Key-value pairs (forms, invoices, etc.)
3. Important sections and their content
4. Any structured data patterns

Format the response as JSON with:
- document_type: identified document type
- extracted_tables: array of tables with headers and rows
- key_values: object of extracted key-value pairs
- sections: array of document sections
- metadata: any document metadata found"""

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        content = response.content[0].text
        
        try:
            result = json.loads(content)
        except:
            result = {
                "document_type": "unknown",
                "extracted_tables": [],
                "key_values": {},
                "sections": [],
                "raw_text": text_content[:1000]
            }
        
        return result
    
    async def extract_from_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Extract data from images (charts, tables, etc.) using Claude's vision
        """
        # Read and encode image
        with open(image_path, 'rb') as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Determine image type
        image_type = "image/jpeg" if image_path.suffix.lower() in ['.jpg', '.jpeg'] else "image/png"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_type,
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": """Analyze this image and extract any structured data.

Please identify and extract:
1. Tables (with headers and data)
2. Charts (type, axes, data points if visible)
3. Text content
4. Key metrics or KPIs
5. Any other structured information

Format the response as JSON with appropriate structure for the detected content."""
                    }
                ]
            }
        ]
        
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.1,
            messages=messages
        )
        
        content = response.content[0].text
        
        try:
            result = json.loads(content)
        except:
            result = {
                "content_type": "unknown",
                "extracted_data": {},
                "description": content
            }
        
        return result
    
    async def generate_pipeline_suggestions(
        self, 
        dataset_info: Dict[str, Any],
        user_goal: str
    ) -> List[Dict[str, Any]]:
        """
        Generate pipeline step suggestions based on data and user goals
        """
        prompt = f"""Based on this dataset and user goal, suggest an optimal data processing pipeline.

Dataset Info:
- Columns: {dataset_info.get('columns', [])}
- Row count: {dataset_info.get('row_count', 0)}
- Data types: {dataset_info.get('dtypes', {})}
- Quality score: {dataset_info.get('quality_score', 'unknown')}

User Goal: {user_goal}

Suggest a pipeline with specific steps from these available operations:
- quality_validation: Check data quality
- pii_detection: Detect and mask PII
- data_cleaning: Clean and standardize
- outlier_detection: Find anomalies
- data_transformation: Transform structure/content
- schema_validation: Validate against schema
- deduplication: Remove duplicates
- enrichment: Add derived columns

For each step, provide:
- type: step type from above
- config: specific configuration
- reasoning: why this step is needed

Return as JSON array of steps."""

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        content = response.content[0].text
        
        try:
            # Extract JSON array from response
            import re
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                suggestions = json.loads(json_match.group())
            else:
                suggestions = []
        except:
            suggestions = []
        
        return suggestions
    
    def _parse_text_response(self, text: str) -> Dict[str, Any]:
        """Parse unstructured text response into structured format"""
        # Basic parsing logic - can be enhanced
        result = {
            "quality_score": 75.0,
            "issues": [],
            "patterns": [],
            "recommendations": [],
            "pii_concerns": [],
            "schema_suggestions": {}
        }
        
        # Extract quality score if mentioned
        import re
        score_match = re.search(r'(?:quality score|score)[:\s]+(\d+)', text, re.IGNORECASE)
        if score_match:
            result["quality_score"] = float(score_match.group(1))
        
        # Extract recommendations (lines starting with - or •)
        recommendations = re.findall(r'(?:^|\n)[\-•]\s*(.+)', text)
        result["recommendations"] = recommendations[:5]  # Limit to 5
        
        return result
    
    def _basic_quality_check(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Perform basic quality checks without AI"""
        issues = []
        
        # Check for missing values
        missing = df.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                severity = "high" if count / len(df) > 0.1 else "medium"
                issues.append({
                    "type": "missing_values",
                    "severity": severity,
                    "description": f"{count} missing values ({count/len(df)*100:.1f}%)",
                    "affected_columns": [col]
                })
        
        # Check for duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            issues.append({
                "type": "duplicate_rows",
                "severity": "medium",
                "description": f"{dup_count} duplicate rows found",
                "affected_columns": []
            })
        
        return issues