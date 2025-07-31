import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
from datetime import datetime
import PyPDF2
import pdf2image
import pytesseract
from PIL import Image
import io

from services.llm.llm_factory import LLMFactory
from services.llm.claude_service import ClaudeService
from core.logging import get_logger

logger = get_logger(__name__)


class PDFProcessor:
    """
    Process PDF documents using Claude Vision for intelligent data extraction
    """
    
    def __init__(self, llm_provider: str = "claude"):
        self.llm_service = LLMFactory.create(llm_provider)
        # Ensure we have Claude for vision capabilities
        if not isinstance(self.llm_service, ClaudeService):
            logger.warning("Non-Claude LLM service may not support vision capabilities")
    
    async def extract_tables(
        self, 
        pdf_path: str,
        confidence_threshold: float = 0.8
    ) -> List[pd.DataFrame]:
        """
        Extract all tables from PDF using Claude Vision
        
        Returns:
            List of DataFrames, one for each detected table
        """
        logger.info(f"Extracting tables from PDF: {pdf_path}")
        
        # Convert PDF pages to images
        images = self._pdf_to_images(pdf_path)
        
        all_tables = []
        
        for page_num, image_path in enumerate(images):
            try:
                # Use Claude Vision to analyze the page
                result = await self.llm_service.analyze_document_with_vision(
                    image_path,
                    analysis_type="extract_tables"
                )
                
                # Process extracted tables
                if "tables" in result:
                    for table_data in result["tables"]:
                        df = self._table_data_to_dataframe(table_data)
                        if df is not None and not df.empty:
                            # Add metadata
                            df.attrs = {
                                "source_pdf": pdf_path,
                                "page_number": page_num + 1,
                                "extraction_confidence": table_data.get("confidence", 1.0),
                                "extraction_timestamp": datetime.utcnow()
                            }
                            all_tables.append(df)
                
            except Exception as e:
                logger.error(f"Failed to process page {page_num + 1}: {str(e)}")
            
            finally:
                # Clean up temporary image
                if os.path.exists(image_path):
                    os.remove(image_path)
        
        logger.info(f"Extracted {len(all_tables)} tables from PDF")
        return all_tables
    
    async def extract_structured_data(
        self,
        pdf_path: str,
        schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract structured data from PDF based on schema or auto-detect
        """
        logger.info(f"Extracting structured data from PDF: {pdf_path}")
        
        # Convert to images
        images = self._pdf_to_images(pdf_path)
        
        structured_data = {
            "source": pdf_path,
            "pages": [],
            "metadata": {},
            "extracted_data": {}
        }
        
        for page_num, image_path in enumerate(images):
            try:
                # Analyze document structure
                result = await self.llm_service.analyze_document_with_vision(
                    image_path,
                    analysis_type="extract_structure"
                )
                
                page_data = {
                    "page_number": page_num + 1,
                    "sections": result.get("sections", []),
                    "key_values": result.get("key_values", {}),
                    "metadata": result.get("metadata", {})
                }
                
                structured_data["pages"].append(page_data)
                
                # Merge key-value pairs
                for key, value in page_data["key_values"].items():
                    if key not in structured_data["extracted_data"]:
                        structured_data["extracted_data"][key] = value
                    elif isinstance(structured_data["extracted_data"][key], list):
                        structured_data["extracted_data"][key].append(value)
                    else:
                        # Convert to list if multiple values
                        structured_data["extracted_data"][key] = [
                            structured_data["extracted_data"][key], value
                        ]
                
            except Exception as e:
                logger.error(f"Failed to process page {page_num + 1}: {str(e)}")
            
            finally:
                if os.path.exists(image_path):
                    os.remove(image_path)
        
        # Apply schema validation if provided
        if schema:
            structured_data = self._validate_against_schema(structured_data, schema)
        
        return structured_data
    
    async def extract_forms(
        self,
        pdf_path: str,
        form_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract form data from fillable or scanned PDFs
        """
        # First try to extract from fillable PDF
        form_data = self._extract_fillable_form_data(pdf_path)
        
        if not form_data:
            # Fall back to vision-based extraction
            logger.info("No fillable form data found, using vision extraction")
            
            images = self._pdf_to_images(pdf_path)
            form_data = {}
            
            for page_num, image_path in enumerate(images):
                try:
                    # Custom prompt for form extraction
                    prompt = """Extract all form fields and their values from this document.
Focus on:
1. Field labels/names
2. Filled values (handwritten or typed)
3. Checkboxes and their states
4. Signature fields

Return as JSON with field_name: field_value pairs."""
                    
                    if form_fields:
                        prompt += f"\n\nSpecifically look for these fields: {', '.join(form_fields)}"
                    
                    result = await self.llm_service.analyze_image(image_path, prompt)
                    
                    # Parse and merge results
                    import json
                    try:
                        page_form_data = json.loads(result.content)
                        form_data.update(page_form_data)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse form data from page {page_num + 1}")
                
                except Exception as e:
                    logger.error(f"Failed to extract form from page {page_num + 1}: {str(e)}")
                
                finally:
                    if os.path.exists(image_path):
                        os.remove(image_path)
        
        return form_data
    
    async def extract_charts_and_graphs(
        self,
        pdf_path: str
    ) -> List[Dict[str, Any]]:
        """
        Extract data from charts and graphs in PDFs
        """
        logger.info(f"Extracting charts from PDF: {pdf_path}")
        
        images = self._pdf_to_images(pdf_path)
        all_charts = []
        
        for page_num, image_path in enumerate(images):
            try:
                result = await self.llm_service.analyze_document_with_vision(
                    image_path,
                    analysis_type="extract_charts"
                )
                
                if "charts" in result:
                    for chart in result["charts"]:
                        chart["page_number"] = page_num + 1
                        chart["source_pdf"] = pdf_path
                        
                        # Convert chart data to DataFrame if possible
                        if "data_points" in chart:
                            chart["dataframe"] = pd.DataFrame(chart["data_points"])
                        
                        all_charts.append(chart)
                
            except Exception as e:
                logger.error(f"Failed to extract charts from page {page_num + 1}: {str(e)}")
            
            finally:
                if os.path.exists(image_path):
                    os.remove(image_path)
        
        return all_charts
    
    def _pdf_to_images(self, pdf_path: str) -> List[str]:
        """Convert PDF pages to images"""
        images = []
        
        try:
            # Convert PDF to images using pdf2image
            pdf_images = pdf2image.convert_from_path(pdf_path, dpi=300)
            
            # Save images temporarily
            temp_dir = tempfile.mkdtemp()
            
            for i, img in enumerate(pdf_images):
                image_path = os.path.join(temp_dir, f"page_{i+1}.png")
                img.save(image_path, "PNG")
                images.append(image_path)
                
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {str(e)}")
            
        return images
    
    def _extract_fillable_form_data(self, pdf_path: str) -> Dict[str, Any]:
        """Extract data from fillable PDF forms"""
        form_data = {}
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                if reader.is_encrypted:
                    reader.decrypt('')
                
                # Extract form fields
                if '/AcroForm' in reader.trailer['/Root']:
                    fields = reader.get_fields()
                    
                    for field_name, field_data in fields.items():
                        value = field_data.get('/V', '')
                        if value:
                            form_data[field_name] = value
                            
        except Exception as e:
            logger.debug(f"No fillable form data found: {str(e)}")
            
        return form_data
    
    def _table_data_to_dataframe(self, table_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Convert extracted table data to DataFrame"""
        try:
            if "headers" in table_data and "rows" in table_data:
                df = pd.DataFrame(table_data["rows"], columns=table_data["headers"])
                
                # Clean and convert data types
                df = self._clean_dataframe(df)
                
                return df
            
            elif "data" in table_data:
                # Alternative format
                return pd.DataFrame(table_data["data"])
                
        except Exception as e:
            logger.error(f"Failed to convert table data to DataFrame: {str(e)}")
            
        return None
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize DataFrame"""
        # Remove empty rows/columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Strip whitespace
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].str.strip()
        
        # Try to infer better data types
        df = df.infer_objects()
        
        # Convert numeric columns
        for col in df.columns:
            # Try numeric conversion
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
            
            # Try datetime conversion
            try:
                if df[col].dtype == 'object':
                    df[col] = pd.to_datetime(df[col], errors='ignore')
            except:
                pass
        
        return df
    
    def _validate_against_schema(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate extracted data against schema"""
        # TODO: Implement schema validation
        return data
    
    async def process_batch(
        self,
        pdf_paths: List[str],
        extraction_type: str = "tables",
        parallel: bool = True
    ) -> Dict[str, Any]:
        """
        Process multiple PDFs in batch
        """
        import asyncio
        
        results = {}
        
        if parallel:
            # Process in parallel
            tasks = []
            for pdf_path in pdf_paths:
                if extraction_type == "tables":
                    task = self.extract_tables(pdf_path)
                elif extraction_type == "structured":
                    task = self.extract_structured_data(pdf_path)
                elif extraction_type == "forms":
                    task = self.extract_forms(pdf_path)
                else:
                    continue
                    
                tasks.append(task)
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for pdf_path, result in zip(pdf_paths, batch_results):
                if isinstance(result, Exception):
                    results[pdf_path] = {"error": str(result)}
                else:
                    results[pdf_path] = result
        else:
            # Process sequentially
            for pdf_path in pdf_paths:
                try:
                    if extraction_type == "tables":
                        result = await self.extract_tables(pdf_path)
                    elif extraction_type == "structured":
                        result = await self.extract_structured_data(pdf_path)
                    elif extraction_type == "forms":
                        result = await self.extract_forms(pdf_path)
                    else:
                        result = {"error": f"Unknown extraction type: {extraction_type}"}
                    
                    results[pdf_path] = result
                    
                except Exception as e:
                    results[pdf_path] = {"error": str(e)}
        
        return results