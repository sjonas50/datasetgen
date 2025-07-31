import pandas as pd
import chardet
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import csv
import io

from .base import BaseConnector, ConnectorConfig, ConnectorResult, DataSchema, SchemaField
from services.llm.llm_factory import LLMFactory
from services.llm.prompt_templates import PromptTemplates
from core.logging import get_logger

logger = get_logger(__name__)


class CSVConnectorConfig(ConnectorConfig):
    """CSV-specific configuration"""
    file_path: str
    delimiter: Optional[str] = None  # Auto-detect if None
    encoding: Optional[str] = None   # Auto-detect if None
    header: Optional[int] = 0
    skip_rows: Optional[List[int]] = None
    na_values: Optional[List[str]] = None
    parse_dates: Optional[Union[bool, List[str]]] = None
    decimal: str = "."
    thousands: Optional[str] = None
    
    # Advanced options
    auto_detect_delimiter: bool = True
    auto_detect_encoding: bool = True
    clean_column_names: bool = True
    infer_date_formats: bool = True


class IntelligentCSVConnector(BaseConnector):
    """CSV connector with LLM-enhanced capabilities"""
    
    def __init__(self, config: Union[Dict[str, Any], CSVConnectorConfig]):
        if isinstance(config, dict):
            config = CSVConnectorConfig(**config)
        super().__init__(config)
        self.config: CSVConnectorConfig = config
        self._detected_params = {}
    
    async def connect(self) -> ConnectorResult:
        """Validate CSV file exists and is readable"""
        try:
            path = Path(self.config.file_path)
            
            if not path.exists():
                return ConnectorResult(
                    success=False,
                    error=f"File not found: {self.config.file_path}"
                )
            
            if not path.is_file():
                return ConnectorResult(
                    success=False,
                    error=f"Path is not a file: {self.config.file_path}"
                )
            
            # Auto-detect parameters if enabled
            if self.config.auto_detect_encoding or self.config.auto_detect_delimiter:
                await self._auto_detect_parameters()
            
            return ConnectorResult(
                success=True,
                metadata={
                    "file_size": path.stat().st_size,
                    "file_name": path.name,
                    "detected_params": self._detected_params
                }
            )
            
        except Exception as e:
            return ConnectorResult(
                success=False,
                error=f"Connection error: {str(e)}"
            )
    
    async def read_data(
        self, 
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> ConnectorResult:
        """Read CSV data with intelligent parsing"""
        try:
            # Prepare read parameters
            read_params = self._prepare_read_params()
            
            # Add row limiting if specified
            if limit:
                read_params["nrows"] = limit
            if offset:
                read_params["skiprows"] = (read_params.get("skiprows", 0) or 0) + offset
            
            # Read CSV
            data = pd.read_csv(self.config.file_path, **read_params)
            
            # Clean column names if enabled
            if self.config.clean_column_names:
                data = self._clean_column_names(data)
            
            # LLM-enhanced processing
            if self.config.llm_enhanced and self.llm_service:
                data = await self._enhance_with_llm(data)
            
            # Infer schema if enabled
            schema = None
            if self.config.auto_schema:
                schema = await self.infer_schema(data)
            
            # Validate data if enabled
            warnings = []
            if self.config.quality_check:
                validation = await self.validate_data(data, schema)
                warnings = validation.get("warnings", [])
            
            return ConnectorResult(
                success=True,
                data=data,
                schema=schema,
                warnings=warnings,
                metadata={
                    "rows_read": len(data),
                    "columns": list(data.columns),
                    "memory_usage": data.memory_usage(deep=True).sum()
                }
            )
            
        except Exception as e:
            logger.error(f"Error reading CSV: {str(e)}")
            return ConnectorResult(
                success=False,
                error=f"Read error: {str(e)}"
            )
    
    async def infer_schema(
        self, 
        data: Optional[pd.DataFrame] = None
    ) -> DataSchema:
        """Infer schema with LLM enhancement"""
        if data is None:
            # Read sample data
            result = await self.read_data(limit=self.config.sample_size)
            if not result.success or result.data is None:
                raise ValueError("Failed to read data for schema inference")
            data = result.data
        
        # Basic schema inference
        fields = []
        
        for col in data.columns:
            field = SchemaField(
                name=col,
                type=self._infer_basic_type(data[col]),
                nullable=data[col].isnull().any(),
                null_count=data[col].isnull().sum(),
                unique_count=data[col].nunique(),
                sample_values=data[col].dropna().head(5).tolist()
            )
            fields.append(field)
        
        schema = DataSchema(
            fields=fields,
            row_count=len(data)
        )
        
        # Enhance with LLM if enabled
        if self.config.llm_enhanced and self.llm_service:
            schema = await self._enhance_schema_with_llm(data, schema)
        
        return schema
    
    async def _auto_detect_parameters(self):
        """Auto-detect CSV parameters"""
        with open(self.config.file_path, 'rb') as f:
            # Read first few KB for detection
            sample = f.read(8192)
        
        # Detect encoding
        if self.config.auto_detect_encoding and not self.config.encoding:
            detected = chardet.detect(sample)
            self.config.encoding = detected['encoding']
            self._detected_params['encoding'] = {
                'value': detected['encoding'],
                'confidence': detected['confidence']
            }
        
        # Detect delimiter
        if self.config.auto_detect_delimiter and not self.config.delimiter:
            try:
                sample_text = sample.decode(self.config.encoding or 'utf-8')
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample_text).delimiter
                self.config.delimiter = delimiter
                self._detected_params['delimiter'] = delimiter
            except Exception as e:
                logger.warning(f"Delimiter detection failed: {str(e)}")
                self.config.delimiter = ','
    
    def _prepare_read_params(self) -> Dict[str, Any]:
        """Prepare pandas read_csv parameters"""
        params = {
            'delimiter': self.config.delimiter or ',',
            'encoding': self.config.encoding or 'utf-8',
            'header': self.config.header,
        }
        
        # Optional parameters
        if self.config.skip_rows:
            params['skiprows'] = self.config.skip_rows
        if self.config.na_values:
            params['na_values'] = self.config.na_values
        if self.config.parse_dates is not None:
            params['parse_dates'] = self.config.parse_dates
        if self.config.thousands:
            params['thousands'] = self.config.thousands
        
        params['decimal'] = self.config.decimal
        
        # Error handling
        params['on_bad_lines'] = 'warn'
        params['engine'] = 'python'  # More flexible but slower
        
        return params
    
    def _clean_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize column names"""
        import re
        
        def clean_name(name):
            # Convert to string
            name = str(name)
            # Replace special characters with underscore
            name = re.sub(r'[^\w\s]', '_', name)
            # Replace spaces with underscore
            name = re.sub(r'\s+', '_', name)
            # Remove consecutive underscores
            name = re.sub(r'_+', '_', name)
            # Strip underscores from ends
            name = name.strip('_')
            # Convert to lowercase
            name = name.lower()
            return name if name else 'column'
        
        # Clean column names
        data.columns = [clean_name(col) for col in data.columns]
        
        # Handle duplicate names
        seen = {}
        new_columns = []
        for col in data.columns:
            if col in seen:
                seen[col] += 1
                new_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_columns.append(col)
        
        data.columns = new_columns
        return data
    
    def _infer_basic_type(self, series: pd.Series) -> str:
        """Infer basic data type from pandas series"""
        dtype = series.dtype
        
        if pd.api.types.is_integer_dtype(dtype):
            return "integer"
        elif pd.api.types.is_float_dtype(dtype):
            return "float"
        elif pd.api.types.is_bool_dtype(dtype):
            return "boolean"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return "datetime"
        elif pd.api.types.is_object_dtype(dtype):
            # Check if it could be datetime
            if self.config.infer_date_formats:
                try:
                    pd.to_datetime(series.dropna().head(100))
                    return "datetime"
                except:
                    pass
            return "string"
        else:
            return "unknown"
    
    async def _enhance_with_llm(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhance data understanding with LLM"""
        # This is where we'd use LLM to:
        # 1. Detect semantic types (email, phone, etc.)
        # 2. Identify data quality issues
        # 3. Suggest transformations
        # 4. Detect relationships
        
        # For now, return data as-is
        # Full implementation would analyze samples and enhance metadata
        return data
    
    async def _enhance_schema_with_llm(
        self, 
        data: pd.DataFrame,
        basic_schema: DataSchema
    ) -> DataSchema:
        """Enhance schema with LLM insights"""
        if not self.llm_service:
            return basic_schema
        
        # Prepare data sample for LLM
        sample_data = data.head(min(100, len(data))).to_dict('records')
        
        # Get LLM schema inference
        try:
            llm_schema = await self.llm_service.infer_schema(sample_data)
            
            # Merge LLM insights with basic schema
            for field in basic_schema.fields:
                llm_field = next(
                    (f for f in llm_schema.get('columns', []) 
                     if f['name'] == field.name),
                    None
                )
                
                if llm_field:
                    # Update with LLM insights
                    field.description = llm_field.get('description', field.description)
                    field.format = llm_field.get('format', field.format)
                    field.constraints = llm_field.get('constraints', field.constraints)
            
            # Add relationships and recommendations
            basic_schema.relationships = llm_schema.get('relationships', [])
            basic_schema.quality_issues = llm_schema.get('quality_issues', [])
            basic_schema.recommendations = llm_schema.get('recommendations', [])
            basic_schema.llm_enhanced = True
            basic_schema.llm_confidence = 0.85  # Would be calculated based on response
            
        except Exception as e:
            logger.warning(f"LLM schema enhancement failed: {str(e)}")
        
        return basic_schema
    
    def get_supported_formats(self) -> List[str]:
        """Get supported file extensions"""
        return ['.csv', '.tsv', '.txt']
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema"""
        base_schema = super().get_config_schema()
        
        csv_schema = {
            "type": "object",
            "required": ["file_path"],
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to CSV file"
                },
                "delimiter": {
                    "type": "string",
                    "description": "Column delimiter (auto-detect if not specified)"
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (auto-detect if not specified)"
                },
                "auto_detect_delimiter": {
                    "type": "boolean",
                    "description": "Automatically detect delimiter",
                    "default": True
                },
                "auto_detect_encoding": {
                    "type": "boolean",
                    "description": "Automatically detect encoding",
                    "default": True
                },
                "clean_column_names": {
                    "type": "boolean",
                    "description": "Clean and standardize column names",
                    "default": True
                }
            }
        }
        
        # Merge schemas
        base_schema["properties"].update(csv_schema["properties"])
        base_schema["required"] = csv_schema.get("required", [])
        
        return base_schema