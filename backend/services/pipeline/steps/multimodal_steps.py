from typing import Dict, Any, List, Optional
import pandas as pd
from pathlib import Path

from services.pipeline.base import PipelineStep, StepInput, StepOutput
from services.multimodal import DocumentParser, PDFProcessor, ImageAnalyzer, FusionEngine
from services.multimodal.fusion_engine import FusionStrategy
from core.logging import get_logger

logger = get_logger(__name__)


class DocumentParseStep(PipelineStep):
    """
    Parse documents of various formats and extract multi-modal data
    """
    
    def __init__(self):
        super().__init__()
        self.parser = DocumentParser()
    
    @property
    def name(self) -> str:
        return "document_parse"
    
    @property
    def description(self) -> str:
        return "Parse documents and extract tables, text, images, and structured data"
    
    async def process(self, input_data: StepInput) -> StepOutput:
        data = input_data.data
        config = input_data.config or {}
        
        # Handle different input formats
        if isinstance(data, str):
            # Single file path
            result = await self.parser.parse_document(data, config)
            return StepOutput(
                data=result,
                metadata={"files_processed": 1}
            )
        
        elif isinstance(data, list):
            # List of file paths
            result = await self.parser.parse_batch(
                data,
                config,
                fusion_strategy=FusionStrategy(config.get("fusion_strategy", "hybrid"))
            )
            return StepOutput(
                data=result,
                metadata={"files_processed": len(data)}
            )
        
        elif isinstance(data, pd.DataFrame) and "file_path" in data.columns:
            # DataFrame with file paths
            file_paths = data["file_path"].tolist()
            result = await self.parser.parse_batch(file_paths, config)
            
            # Merge results back to DataFrame
            parsed_data = []
            for idx, row in data.iterrows():
                file_path = row["file_path"]
                if file_path in result.get("individual_documents", {}):
                    parsed = result["individual_documents"][file_path]
                    row_data = row.to_dict()
                    row_data["parsed_data"] = parsed
                    parsed_data.append(row_data)
            
            return StepOutput(
                data=pd.DataFrame(parsed_data),
                metadata={
                    "files_processed": len(file_paths),
                    "fusion_applied": "fused_data" in result
                }
            )
        
        else:
            raise ValueError(f"Unsupported input type: {type(data)}")


class PDFTableExtractStep(PipelineStep):
    """
    Extract tables from PDF documents using Claude Vision
    """
    
    def __init__(self):
        super().__init__()
        self.processor = PDFProcessor()
    
    @property
    def name(self) -> str:
        return "pdf_table_extract"
    
    @property
    def description(self) -> str:
        return "Extract tables from PDF documents with high accuracy"
    
    async def process(self, input_data: StepInput) -> StepOutput:
        data = input_data.data
        config = input_data.config or {}
        
        all_tables = []
        metadata = {
            "pdfs_processed": 0,
            "tables_extracted": 0,
            "pages_analyzed": 0
        }
        
        # Get PDF paths
        pdf_paths = []
        if isinstance(data, str):
            pdf_paths = [data]
        elif isinstance(data, list):
            pdf_paths = [p for p in data if p.lower().endswith('.pdf')]
        elif isinstance(data, pd.DataFrame) and "file_path" in data.columns:
            pdf_paths = data[data["file_path"].str.lower().str.endswith('.pdf')]["file_path"].tolist()
        
        # Process each PDF
        for pdf_path in pdf_paths:
            try:
                tables = await self.processor.extract_tables(
                    pdf_path,
                    confidence_threshold=config.get("confidence_threshold", 0.8)
                )
                
                all_tables.extend(tables)
                metadata["pdfs_processed"] += 1
                metadata["tables_extracted"] += len(tables)
                
            except Exception as e:
                logger.error(f"Failed to extract tables from {pdf_path}: {str(e)}")
        
        # Combine tables if requested
        if config.get("combine_tables", False) and all_tables:
            combined = pd.concat(all_tables, ignore_index=True)
            return StepOutput(
                data=combined,
                metadata=metadata
            )
        else:
            return StepOutput(
                data=all_tables,
                metadata=metadata
            )


class ImageDataExtractStep(PipelineStep):
    """
    Extract data from images including charts, tables, and dashboards
    """
    
    def __init__(self):
        super().__init__()
        self.analyzer = ImageAnalyzer()
    
    @property
    def name(self) -> str:
        return "image_data_extract"
    
    @property
    def description(self) -> str:
        return "Extract structured data from images, charts, and dashboards"
    
    async def process(self, input_data: StepInput) -> StepOutput:
        data = input_data.data
        config = input_data.config or {}
        
        analysis_type = config.get("analysis_type", "auto")
        results = {}
        
        # Get image paths
        image_paths = []
        if isinstance(data, str):
            image_paths = [data]
        elif isinstance(data, list):
            image_paths = data
        elif isinstance(data, pd.DataFrame) and "file_path" in data.columns:
            image_paths = data["file_path"].tolist()
        
        # Filter for image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
        image_paths = [
            p for p in image_paths 
            if Path(p).suffix.lower() in image_extensions
        ]
        
        # Auto-detect analysis type if needed
        if analysis_type == "auto":
            # Batch analyze all images
            for img_path in image_paths:
                try:
                    # Try different analysis types
                    if "dashboard" in img_path.lower():
                        result = await self.analyzer.analyze_dashboard_screenshot(img_path)
                    elif "chart" in img_path.lower() or "graph" in img_path.lower():
                        result = await self.analyzer.extract_chart_data(img_path)
                    elif "table" in img_path.lower():
                        result = await self.analyzer.extract_table_from_image(img_path)
                    else:
                        # Try chart extraction by default
                        result = await self.analyzer.extract_chart_data(img_path)
                    
                    results[img_path] = result
                    
                except Exception as e:
                    logger.error(f"Failed to analyze {img_path}: {str(e)}")
                    results[img_path] = {"error": str(e)}
        else:
            # Use specified analysis type
            batch_results = await self.analyzer.batch_analyze_images(
                image_paths,
                analysis_type
            )
            results = batch_results
        
        return StepOutput(
            data=results,
            metadata={
                "images_processed": len(image_paths),
                "successful_extractions": sum(1 for r in results.values() if "error" not in r)
            }
        )


class MultiModalFusionStep(PipelineStep):
    """
    Fuse data from multiple modalities into unified dataset
    """
    
    def __init__(self):
        super().__init__()
        self.fusion_engine = FusionEngine()
    
    @property
    def name(self) -> str:
        return "multimodal_fusion"
    
    @property
    def description(self) -> str:
        return "Intelligently fuse data from multiple modalities"
    
    async def process(self, input_data: StepInput) -> StepOutput:
        data = input_data.data
        config = input_data.config or {}
        
        # Extract fusion strategy
        strategy = FusionStrategy(config.get("fusion_strategy", "hybrid"))
        target_schema = config.get("target_schema")
        
        # Prepare modalities
        modalities = {}
        
        if isinstance(data, dict):
            # Direct modality dictionary
            modalities = data
        elif isinstance(data, list):
            # List of DataFrames or other data
            for i, item in enumerate(data):
                if isinstance(item, pd.DataFrame):
                    modalities[f"modality_{i}"] = {
                        "data": item,
                        "type": "tabular"
                    }
                else:
                    modalities[f"modality_{i}"] = {
                        "data": item,
                        "type": self._infer_modality_type(item)
                    }
        else:
            raise ValueError("Input must be dict of modalities or list of data")
        
        # Apply fusion
        fused_data = await self.fusion_engine.fuse_modalities(
            modalities,
            strategy=strategy,
            target_schema=target_schema
        )
        
        return StepOutput(
            data=fused_data,
            metadata={
                "modalities_fused": len(modalities),
                "fusion_strategy": strategy.value,
                "quality_score": fused_data.attrs.get("fusion_quality", {})
            }
        )
    
    def _infer_modality_type(self, data: Any) -> str:
        """Infer modality type from data"""
        if isinstance(data, pd.DataFrame):
            return "tabular"
        elif isinstance(data, (str, list)) and isinstance(data, str) or (isinstance(data, list) and all(isinstance(x, str) for x in data)):
            return "text"
        elif isinstance(data, dict) and "image" in str(data).lower():
            return "image"
        else:
            return "tabular"


class DocumentQualityCheckStep(PipelineStep):
    """
    Assess quality of document parsing and extraction
    """
    
    def __init__(self):
        super().__init__()
        self.parser = DocumentParser()
    
    @property
    def name(self) -> str:
        return "document_quality_check"
    
    @property
    def description(self) -> str:
        return "Assess extraction quality and completeness"
    
    async def process(self, input_data: StepInput) -> StepOutput:
        data = input_data.data
        config = input_data.config or {}
        
        quality_threshold = config.get("quality_threshold", 0.7)
        quality_reports = []
        failed_docs = []
        
        # Handle parsed document data
        if isinstance(data, dict) and "individual_documents" in data:
            # From batch parsing
            for file_path, parsed_data in data["individual_documents"].items():
                quality = await self.parser.analyze_document_quality(
                    file_path,
                    parsed_data
                )
                
                quality["file_path"] = file_path
                quality_reports.append(quality)
                
                if quality["overall_score"] < quality_threshold:
                    failed_docs.append(file_path)
        
        elif isinstance(data, pd.DataFrame) and "parsed_data" in data.columns:
            # DataFrame with parsed data
            for idx, row in data.iterrows():
                if "file_path" in row and "parsed_data" in row:
                    quality = await self.parser.analyze_document_quality(
                        row["file_path"],
                        row["parsed_data"]
                    )
                    
                    quality["file_path"] = row["file_path"]
                    quality_reports.append(quality)
                    
                    if quality["overall_score"] < quality_threshold:
                        failed_docs.append(row["file_path"])
        
        # Create quality DataFrame
        quality_df = pd.DataFrame(quality_reports)
        
        return StepOutput(
            data=quality_df,
            metadata={
                "documents_checked": len(quality_reports),
                "passed_threshold": len(quality_reports) - len(failed_docs),
                "failed_documents": failed_docs,
                "average_quality_score": quality_df["overall_score"].mean() if not quality_df.empty else 0
            }
        )


# Export all multi-modal steps
__all__ = [
    "DocumentParseStep",
    "PDFTableExtractStep",
    "ImageDataExtractStep",
    "MultiModalFusionStep",
    "DocumentQualityCheckStep"
]