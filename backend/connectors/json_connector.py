import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import jsonpath_ng
from jsonpath_ng import parse

from .base import BaseConnector, ConnectorConfig, ConnectorResult, DataSchema, SchemaField
from services.llm.llm_factory import LLMFactory
from core.logging import get_logger

logger = get_logger(__name__)


class JSONConnectorConfig(ConnectorConfig):
    """JSON-specific configuration"""
    file_path: str
    json_path: Optional[str] = None  # JSONPath expression to extract data
    orient: str = "records"  # pandas JSON orientation
    lines: bool = False  # True for JSONL files
    encoding: str = "utf-8"
    flatten_nested: bool = True
    max_level: int = 3  # Max nesting level to flatten
    
    # Array handling
    array_handling: str = "expand"  # expand, join, or first
    array_separator: str = ", "
    
    # Schema options
    infer_schema_from_all: bool = True  # Use all records for schema
    handle_mixed_types: bool = True


class IntelligentJSONConnector(BaseConnector):
    """JSON connector with LLM-enhanced capabilities"""
    
    def __init__(self, config: Union[Dict[str, Any], JSONConnectorConfig]):
        if isinstance(config, dict):
            config = JSONConnectorConfig(**config)
        super().__init__(config)
        self.config: JSONConnectorConfig = config
        self._raw_data = None
    
    async def connect(self) -> ConnectorResult:
        """Validate JSON file exists and is readable"""
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
            
            # Validate JSON structure
            validation = await self._validate_json_structure()
            if not validation["valid"]:
                return ConnectorResult(
                    success=False,
                    error=validation["error"]
                )
            
            return ConnectorResult(
                success=True,
                metadata={
                    "file_size": path.stat().st_size,
                    "file_name": path.name,
                    "json_structure": validation["structure"]
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
        """Read JSON data with intelligent parsing"""
        try:
            # Read JSON file
            if self.config.lines:
                data = await self._read_jsonl(limit, offset)
            else:
                data = await self._read_json(limit, offset)
            
            # Flatten nested structures if enabled
            if self.config.flatten_nested:
                data = self._flatten_dataframe(data)
            
            # Handle mixed types if enabled
            if self.config.handle_mixed_types:
                data = self._handle_mixed_types(data)
            
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
            logger.error(f"Error reading JSON: {str(e)}")
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
        
        # If we have raw JSON and LLM is enabled, analyze structure
        if self._raw_data and self.config.llm_enhanced and self.llm_service:
            return await self._infer_schema_from_json()
        
        # Otherwise, infer from DataFrame
        fields = []
        
        for col in data.columns:
            # Detect if column contains JSON strings
            is_json_column = False
            if data[col].dtype == 'object':
                try:
                    sample = data[col].dropna().iloc[0] if len(data[col].dropna()) > 0 else None
                    if sample and isinstance(sample, str):
                        json.loads(sample)
                        is_json_column = True
                except:
                    pass
            
            field = SchemaField(
                name=col,
                type="object" if is_json_column else self._infer_type(data[col]),
                nullable=data[col].isnull().any(),
                null_count=data[col].isnull().sum(),
                unique_count=data[col].nunique(),
                sample_values=data[col].dropna().head(5).tolist()
            )
            
            # Add format hints for nested columns
            if '.' in col:
                field.description = f"Nested field from: {col.split('.')[0]}"
            
            fields.append(field)
        
        schema = DataSchema(
            fields=fields,
            row_count=len(data)
        )
        
        # Enhance with LLM if enabled
        if self.config.llm_enhanced and self.llm_service:
            schema = await self._enhance_schema_with_llm(data, schema)
        
        return schema
    
    async def _validate_json_structure(self) -> Dict[str, Any]:
        """Validate and analyze JSON structure"""
        try:
            with open(self.config.file_path, 'r', encoding=self.config.encoding) as f:
                if self.config.lines:
                    # Validate JSONL
                    lines = f.readlines()
                    for i, line in enumerate(lines[:10]):  # Check first 10 lines
                        json.loads(line)
                    
                    structure = {
                        "type": "jsonl",
                        "lines": len(lines),
                        "sample": json.loads(lines[0]) if lines else {}
                    }
                else:
                    # Validate regular JSON
                    content = f.read()
                    data = json.loads(content)
                    self._raw_data = data
                    
                    structure = {
                        "type": self._detect_json_structure(data),
                        "sample": self._get_sample(data)
                    }
            
            return {
                "valid": True,
                "structure": structure
            }
            
        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "error": f"Invalid JSON: {str(e)}"
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation error: {str(e)}"
            }
    
    async def _read_json(
        self, 
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> pd.DataFrame:
        """Read regular JSON file"""
        with open(self.config.file_path, 'r', encoding=self.config.encoding) as f:
            data = json.load(f)
        
        # Apply JSONPath if specified
        if self.config.json_path:
            jsonpath_expr = parse(self.config.json_path)
            matches = jsonpath_expr.find(data)
            data = [match.value for match in matches]
        
        # Convert to DataFrame
        if isinstance(data, list):
            # Array of objects
            if offset:
                data = data[offset:]
            if limit:
                data = data[:limit]
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # Single object or nested structure
            if self._is_record_dict(data):
                # Dictionary of records
                df = pd.DataFrame.from_dict(data, orient='index')
            else:
                # Single record
                df = pd.DataFrame([data])
        else:
            raise ValueError(f"Unsupported JSON structure: {type(data)}")
        
        return df
    
    async def _read_jsonl(
        self, 
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> pd.DataFrame:
        """Read JSONL (JSON Lines) file"""
        records = []
        
        with open(self.config.file_path, 'r', encoding=self.config.encoding) as f:
            for i, line in enumerate(f):
                if offset and i < offset:
                    continue
                if limit and len(records) >= limit:
                    break
                
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping line {i}: {str(e)}")
        
        return pd.DataFrame(records)
    
    def _flatten_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten nested structures in DataFrame"""
        if df.empty:
            return df
        
        # Identify columns with nested data
        nested_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                if isinstance(sample, (dict, list)):
                    nested_cols.append(col)
        
        if not nested_cols:
            return df
        
        # Flatten nested columns
        for col in nested_cols:
            if df[col].apply(lambda x: isinstance(x, dict)).any():
                # Flatten dictionaries
                normalized = pd.json_normalize(df[col].dropna())
                normalized.columns = [f"{col}.{subcol}" for subcol in normalized.columns]
                normalized.index = df[col].dropna().index
                
                # Drop original column and join normalized
                df = df.drop(columns=[col]).join(normalized)
            
            elif df[col].apply(lambda x: isinstance(x, list)).any():
                # Handle arrays based on config
                if self.config.array_handling == "expand":
                    # Expand arrays into separate rows
                    df = df.explode(col)
                elif self.config.array_handling == "join":
                    # Join array elements
                    df[col] = df[col].apply(
                        lambda x: self.config.array_separator.join(map(str, x)) 
                        if isinstance(x, list) else x
                    )
                elif self.config.array_handling == "first":
                    # Take first element
                    df[col] = df[col].apply(
                        lambda x: x[0] if isinstance(x, list) and x else x
                    )
        
        return df
    
    def _handle_mixed_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle columns with mixed types"""
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to infer better type
                try:
                    # Try numeric conversion
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
                
                # Try datetime conversion
                if df[col].dtype == 'object':
                    try:
                        temp = pd.to_datetime(df[col], errors='coerce')
                        if temp.notna().sum() > len(df) * 0.5:  # More than 50% valid
                            df[col] = temp
                    except:
                        pass
        
        return df
    
    def _detect_json_structure(self, data: Any) -> str:
        """Detect the structure type of JSON data"""
        if isinstance(data, list):
            return "array"
        elif isinstance(data, dict):
            if all(isinstance(v, dict) for v in data.values()):
                return "object_of_objects"
            elif all(isinstance(v, list) for v in data.values()):
                return "object_of_arrays"
            else:
                return "object"
        else:
            return "primitive"
    
    def _is_record_dict(self, data: dict) -> bool:
        """Check if dictionary represents multiple records"""
        if len(data) < 2:
            return False
        
        # Check if all values have similar structure
        values = list(data.values())
        if all(isinstance(v, dict) for v in values):
            # Check if dictionaries have similar keys
            keys_sets = [set(v.keys()) for v in values[:5]]  # Check first 5
            return all(keys == keys_sets[0] for keys in keys_sets)
        
        return False
    
    def _get_sample(self, data: Any, max_items: int = 3) -> Any:
        """Get a sample of the data structure"""
        if isinstance(data, list):
            return data[:max_items]
        elif isinstance(data, dict):
            items = list(data.items())[:max_items]
            return dict(items)
        else:
            return data
    
    def _infer_type(self, series: pd.Series) -> str:
        """Infer data type from pandas series"""
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
            # Check for nested structures
            sample = series.dropna().iloc[0] if len(series.dropna()) > 0 else None
            if isinstance(sample, dict):
                return "object"
            elif isinstance(sample, list):
                return "array"
            else:
                return "string"
        else:
            return "unknown"
    
    async def _infer_schema_from_json(self) -> DataSchema:
        """Infer schema directly from JSON structure using LLM"""
        if not self.llm_service or not self._raw_data:
            return None
        
        # Get LLM to analyze the JSON structure
        sample = self._get_sample(self._raw_data, max_items=10)
        
        try:
            llm_schema = await self.llm_service.infer_schema(sample)
            
            # Convert LLM response to DataSchema
            fields = []
            for col_info in llm_schema.get('columns', []):
                field = SchemaField(
                    name=col_info['name'],
                    type=col_info['type'],
                    nullable=col_info.get('nullable', True),
                    description=col_info.get('description'),
                    format=col_info.get('format'),
                    constraints=col_info.get('constraints', {})
                )
                fields.append(field)
            
            schema = DataSchema(
                fields=fields,
                row_count=len(self._raw_data) if isinstance(self._raw_data, list) else 1,
                relationships=llm_schema.get('relationships', []),
                quality_issues=llm_schema.get('quality_issues', []),
                recommendations=llm_schema.get('recommendations', []),
                llm_enhanced=True,
                llm_confidence=0.9
            )
            
            return schema
            
        except Exception as e:
            logger.warning(f"LLM schema inference failed: {str(e)}")
            return None
    
    async def _enhance_with_llm(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhance data understanding with LLM"""
        # Implementation would analyze data patterns and suggest improvements
        return data
    
    async def _enhance_schema_with_llm(
        self, 
        data: pd.DataFrame,
        basic_schema: DataSchema
    ) -> DataSchema:
        """Enhance schema with LLM insights"""
        if not self.llm_service:
            return basic_schema
        
        # Similar to CSV connector implementation
        return basic_schema
    
    def get_supported_formats(self) -> List[str]:
        """Get supported file extensions"""
        return ['.json', '.jsonl']
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema"""
        base_schema = super().get_config_schema()
        
        json_schema = {
            "type": "object",
            "required": ["file_path"],
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to JSON file"
                },
                "json_path": {
                    "type": "string",
                    "description": "JSONPath expression to extract data"
                },
                "lines": {
                    "type": "boolean",
                    "description": "True for JSONL (JSON Lines) format",
                    "default": False
                },
                "flatten_nested": {
                    "type": "boolean",
                    "description": "Flatten nested JSON structures",
                    "default": True
                },
                "array_handling": {
                    "type": "string",
                    "enum": ["expand", "join", "first"],
                    "description": "How to handle array fields",
                    "default": "expand"
                }
            }
        }
        
        base_schema["properties"].update(json_schema["properties"])
        base_schema["required"] = json_schema.get("required", [])
        
        return base_schema