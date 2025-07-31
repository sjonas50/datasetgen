import os
import json
from typing import Dict, Any, List, Optional, Union
import pandas as pd
from datetime import datetime
import mimetypes
from pathlib import Path

from services.llm.llm_factory import LLMFactory
from services.multimodal.pdf_processor import PDFProcessor
from services.multimodal.image_analyzer import ImageAnalyzer
from services.multimodal.fusion_engine import FusionEngine, FusionStrategy, ModalityType
from core.logging import get_logger

logger = get_logger(__name__)


class DocumentParser:
    """
    Unified document parser that handles various document formats
    and extracts multi-modal data using appropriate processors
    """
    
    def __init__(self, llm_provider: str = "claude"):
        self.llm_service = LLMFactory.create(llm_provider)
        self.pdf_processor = PDFProcessor(llm_provider)
        self.image_analyzer = ImageAnalyzer(llm_provider)
        self.fusion_engine = FusionEngine(llm_provider)
        
        # Supported document types and their processors
        self.processors = {
            ".pdf": self._process_pdf,
            ".png": self._process_image,
            ".jpg": self._process_image,
            ".jpeg": self._process_image,
            ".gif": self._process_image,
            ".bmp": self._process_image,
            ".tiff": self._process_image,
            ".txt": self._process_text,
            ".csv": self._process_csv,
            ".json": self._process_json,
            ".xlsx": self._process_excel,
            ".xls": self._process_excel,
            ".docx": self._process_docx,
            ".pptx": self._process_pptx
        }
    
    async def parse_document(
        self,
        file_path: str,
        extraction_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Parse any supported document and extract multi-modal data
        
        Args:
            file_path: Path to the document
            extraction_config: Configuration for extraction behavior
            
        Returns:
            Dictionary containing extracted data and metadata
        """
        logger.info(f"Parsing document: {file_path}")
        
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Determine file type
        file_ext = Path(file_path).suffix.lower()
        mime_type = mimetypes.guess_type(file_path)[0]
        
        # Get appropriate processor
        processor = self.processors.get(file_ext)
        if not processor:
            # Try to process as text if unknown type
            logger.warning(f"Unknown file type {file_ext}, attempting text extraction")
            processor = self._process_text
        
        # Configure extraction
        config = extraction_config or {}
        config.setdefault("extract_tables", True)
        config.setdefault("extract_images", True)
        config.setdefault("extract_text", True)
        config.setdefault("extract_metadata", True)
        
        # Process document
        try:
            result = await processor(file_path, config)
            
            # Add common metadata
            result["metadata"] = result.get("metadata", {})
            result["metadata"].update({
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "file_size": os.path.getsize(file_path),
                "file_type": file_ext,
                "mime_type": mime_type,
                "processed_at": datetime.utcnow().isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse document: {str(e)}")
            raise
    
    async def parse_batch(
        self,
        file_paths: List[str],
        extraction_config: Optional[Dict[str, Any]] = None,
        fusion_strategy: FusionStrategy = FusionStrategy.HYBRID
    ) -> Dict[str, Any]:
        """
        Parse multiple documents and optionally fuse the results
        """
        import asyncio
        
        # Parse all documents in parallel
        tasks = [
            self.parse_document(path, extraction_config)
            for path in file_paths
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successful parses and errors
        parsed_docs = {}
        errors = {}
        
        for path, result in zip(file_paths, results):
            if isinstance(result, Exception):
                errors[path] = str(result)
            else:
                parsed_docs[path] = result
        
        # Prepare fusion if multiple successful parses
        if len(parsed_docs) > 1:
            # Extract modalities for fusion
            modalities = {}
            
            for path, doc_data in parsed_docs.items():
                # Tables
                if "tables" in doc_data and doc_data["tables"]:
                    modalities[f"{path}_tables"] = {
                        "data": pd.concat(doc_data["tables"]) if len(doc_data["tables"]) > 1 else doc_data["tables"][0],
                        "type": ModalityType.TABULAR
                    }
                
                # Text content
                if "text" in doc_data:
                    modalities[f"{path}_text"] = {
                        "data": doc_data["text"],
                        "type": ModalityType.TEXT
                    }
                
                # Structured data
                if "structured_data" in doc_data:
                    df = pd.DataFrame([doc_data["structured_data"]])
                    modalities[f"{path}_structured"] = {
                        "data": df,
                        "type": ModalityType.TABULAR
                    }
            
            # Fuse modalities if requested
            if modalities:
                try:
                    fused_data = await self.fusion_engine.fuse_modalities(
                        modalities,
                        strategy=fusion_strategy
                    )
                    
                    return {
                        "individual_documents": parsed_docs,
                        "fused_data": fused_data,
                        "fusion_strategy": fusion_strategy.value,
                        "errors": errors
                    }
                except Exception as e:
                    logger.error(f"Fusion failed: {str(e)}")
        
        return {
            "individual_documents": parsed_docs,
            "errors": errors
        }
    
    async def _process_pdf(self, file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process PDF documents
        """
        result = {
            "type": "pdf",
            "tables": [],
            "charts": [],
            "text": "",
            "forms": {},
            "structured_data": {}
        }
        
        # Extract tables
        if config.get("extract_tables"):
            try:
                tables = await self.pdf_processor.extract_tables(file_path)
                result["tables"] = tables
            except Exception as e:
                logger.error(f"Table extraction failed: {str(e)}")
        
        # Extract charts
        if config.get("extract_images"):
            try:
                charts = await self.pdf_processor.extract_charts_and_graphs(file_path)
                result["charts"] = charts
            except Exception as e:
                logger.error(f"Chart extraction failed: {str(e)}")
        
        # Extract structured data
        if config.get("extract_text"):
            try:
                structured = await self.pdf_processor.extract_structured_data(file_path)
                result["structured_data"] = structured.get("extracted_data", {})
                result["text"] = self._extract_text_from_structured(structured)
            except Exception as e:
                logger.error(f"Structured extraction failed: {str(e)}")
        
        # Extract forms
        if config.get("extract_forms"):
            try:
                forms = await self.pdf_processor.extract_forms(file_path)
                result["forms"] = forms
            except Exception as e:
                logger.error(f"Form extraction failed: {str(e)}")
        
        return result
    
    async def _process_image(self, file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image files
        """
        result = {
            "type": "image",
            "tables": [],
            "charts": [],
            "text_regions": {}
        }
        
        # Check if image contains table
        if config.get("extract_tables"):
            try:
                table_df = await self.image_analyzer.extract_table_from_image(file_path)
                if not table_df.empty:
                    result["tables"] = [table_df]
            except Exception as e:
                logger.error(f"Table extraction from image failed: {str(e)}")
        
        # Check if image contains chart
        if config.get("extract_images"):
            try:
                chart_data = await self.image_analyzer.extract_chart_data(file_path)
                if chart_data:
                    result["charts"] = [chart_data]
            except Exception as e:
                logger.error(f"Chart extraction from image failed: {str(e)}")
        
        # Extract text regions
        if config.get("extract_text"):
            try:
                text_regions = await self.image_analyzer.detect_and_extract_text_regions(file_path)
                result["text_regions"] = text_regions
                result["text"] = " ".join(
                    text for texts in text_regions.values() for text in texts
                )
            except Exception as e:
                logger.error(f"Text extraction from image failed: {str(e)}")
        
        # Check if it's a dashboard
        if "dashboard" in file_path.lower() or config.get("is_dashboard"):
            try:
                dashboard_data = await self.image_analyzer.analyze_dashboard_screenshot(file_path)
                result["dashboard_metrics"] = dashboard_data
            except Exception as e:
                logger.error(f"Dashboard analysis failed: {str(e)}")
        
        return result
    
    async def _process_text(self, file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process plain text files
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        result = {
            "type": "text",
            "text": content,
            "tables": [],
            "structured_data": {}
        }
        
        # Try to extract structured data from text
        if config.get("extract_text"):
            structured = await self._extract_structured_from_text(content)
            result["structured_data"] = structured
        
        return result
    
    async def _process_csv(self, file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process CSV files
        """
        try:
            # Use pandas to read CSV with auto-detection
            df = pd.read_csv(file_path, encoding_errors='ignore')
            
            return {
                "type": "csv",
                "tables": [df],
                "text": "",
                "metadata": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist()
                }
            }
        except Exception as e:
            logger.error(f"CSV processing failed: {str(e)}")
            raise
    
    async def _process_json(self, file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process JSON files
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        result = {
            "type": "json",
            "structured_data": data,
            "tables": [],
            "text": json.dumps(data, indent=2)
        }
        
        # Try to convert to tabular if possible
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            df = pd.DataFrame(data)
            result["tables"] = [df]
        elif isinstance(data, dict):
            # Try to find tabular data within
            for key, value in data.items():
                if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                    df = pd.DataFrame(value)
                    df.attrs["source_key"] = key
                    result["tables"].append(df)
        
        return result
    
    async def _process_excel(self, file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Excel files
        """
        result = {
            "type": "excel",
            "tables": [],
            "sheets": {},
            "text": ""
        }
        
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                df.attrs["sheet_name"] = sheet_name
                result["tables"].append(df)
                result["sheets"][sheet_name] = {
                    "rows": len(df),
                    "columns": len(df.columns)
                }
            
            result["metadata"] = {
                "sheet_count": len(excel_file.sheet_names),
                "sheet_names": excel_file.sheet_names
            }
            
        except Exception as e:
            logger.error(f"Excel processing failed: {str(e)}")
            raise
        
        return result
    
    async def _process_docx(self, file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Word documents
        """
        # For now, use LLM to extract content
        # In production, would use python-docx
        prompt = """Extract all content from this Word document including:
1. All text paragraphs
2. Any tables (convert to structured format)
3. Headers and footers
4. Any embedded data

Return as JSON with sections for text, tables, and metadata."""
        
        result = await self.llm_service.complete(prompt + f"\n\nFile: {file_path}")
        
        return {
            "type": "docx",
            "text": result.content,
            "tables": [],
            "structured_data": {}
        }
    
    async def _process_pptx(self, file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process PowerPoint presentations
        """
        # For now, use LLM to extract content
        # In production, would use python-pptx
        prompt = """Extract all content from this PowerPoint presentation including:
1. Text from all slides
2. Any tables or data
3. Speaker notes
4. Slide titles and structure

Return as JSON with sections for each slide."""
        
        result = await self.llm_service.complete(prompt + f"\n\nFile: {file_path}")
        
        return {
            "type": "pptx",
            "text": result.content,
            "slides": [],
            "structured_data": {}
        }
    
    def _extract_text_from_structured(self, structured_data: Dict[str, Any]) -> str:
        """
        Extract text content from structured data
        """
        text_parts = []
        
        for page in structured_data.get("pages", []):
            # Extract text from sections
            for section in page.get("sections", []):
                if "text" in section:
                    text_parts.append(section["text"])
            
            # Extract text from key-value pairs
            for key, value in page.get("key_values", {}).items():
                text_parts.append(f"{key}: {value}")
        
        return "\n".join(text_parts)
    
    async def _extract_structured_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract structured data from plain text
        """
        prompt = f"""Extract any structured information from this text:

{text[:2000]}  # Limit to first 2000 chars

Look for:
1. Key-value pairs
2. Lists or enumerations  
3. Tabular data
4. Dates and numbers
5. Named entities

Return as JSON."""
        
        result = await self.llm_service.complete(prompt)
        
        try:
            return json.loads(result.content)
        except:
            return {"raw_text": text[:1000]}
    
    async def analyze_document_quality(
        self,
        file_path: str,
        parsed_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze document quality and extraction confidence
        """
        if not parsed_data:
            parsed_data = await self.parse_document(file_path)
        
        quality_metrics = {
            "completeness": 0.0,
            "accuracy_confidence": 0.0,
            "structure_quality": 0.0,
            "data_richness": 0.0,
            "issues": []
        }
        
        # Check completeness
        expected_fields = ["tables", "text", "structured_data"]
        present_fields = sum(1 for field in expected_fields if field in parsed_data and parsed_data[field])
        quality_metrics["completeness"] = present_fields / len(expected_fields)
        
        # Check data richness
        table_count = len(parsed_data.get("tables", []))
        text_length = len(parsed_data.get("text", ""))
        structured_keys = len(parsed_data.get("structured_data", {}))
        
        if table_count > 0:
            quality_metrics["data_richness"] += 0.4
        if text_length > 100:
            quality_metrics["data_richness"] += 0.3
        if structured_keys > 5:
            quality_metrics["data_richness"] += 0.3
        
        # Overall quality score
        quality_metrics["overall_score"] = sum([
            quality_metrics["completeness"] * 0.3,
            quality_metrics["data_richness"] * 0.4,
            0.3  # Base confidence
        ])
        
        return quality_metrics