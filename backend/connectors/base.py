from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel
import pandas as pd
from datetime import datetime

from core.logging import get_logger

logger = get_logger(__name__)


class ConnectorConfig(BaseModel):
    """Base configuration for connectors"""
    source_type: str
    llm_enhanced: bool = True
    auto_schema: bool = True
    quality_check: bool = True
    sample_size: int = 1000  # Rows to sample for schema inference
    
    # LLM options
    llm_provider: str = "claude"
    llm_model: Optional[str] = None
    
    # Additional config
    extra_config: Dict[str, Any] = {}


class SchemaField(BaseModel):
    """Schema field definition"""
    name: str
    type: str  # string, integer, float, boolean, datetime, object, array
    nullable: bool = True
    description: Optional[str] = None
    format: Optional[str] = None  # email, phone, url, date, etc.
    constraints: Dict[str, Any] = {}  # min, max, pattern, enum, etc.
    
    # Quality metrics
    null_count: int = 0
    unique_count: int = 0
    sample_values: List[Any] = []


class DataSchema(BaseModel):
    """Complete data schema"""
    fields: List[SchemaField]
    row_count: int
    relationships: List[Dict[str, Any]] = []
    quality_issues: List[Dict[str, Any]] = []
    recommendations: List[str] = []
    inferred_at: datetime = datetime.utcnow()
    
    # LLM metadata
    llm_enhanced: bool = False
    llm_confidence: float = 0.0


class ConnectorResult(BaseModel):
    """Result from connector operations"""
    success: bool
    data: Optional[pd.DataFrame] = None
    schema: Optional[DataSchema] = None
    error: Optional[str] = None
    warnings: List[str] = []
    metadata: Dict[str, Any] = {}
    
    class Config:
        arbitrary_types_allowed = True


class BaseConnector(ABC):
    """Abstract base class for data connectors"""
    
    def __init__(self, config: ConnectorConfig):
        self.config = config
        self.llm_service = None
        
        if config.llm_enhanced:
            self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM service if enhanced mode is enabled"""
        from services.llm.llm_factory import LLMFactory
        
        try:
            self.llm_service = LLMFactory.create(
                provider=self.config.llm_provider,
                model=self.config.llm_model
            )
        except Exception as e:
            logger.warning(f"Failed to initialize LLM service: {str(e)}")
            self.config.llm_enhanced = False
    
    @abstractmethod
    async def connect(self) -> ConnectorResult:
        """Establish connection to data source"""
        pass
    
    @abstractmethod
    async def read_data(
        self, 
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> ConnectorResult:
        """Read data from source"""
        pass
    
    @abstractmethod
    async def infer_schema(
        self, 
        data: Optional[pd.DataFrame] = None
    ) -> DataSchema:
        """Infer schema from data"""
        pass
    
    async def validate_data(
        self, 
        data: pd.DataFrame,
        schema: Optional[DataSchema] = None
    ) -> Dict[str, Any]:
        """Validate data quality"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        # Basic statistics
        validation_results["statistics"] = {
            "row_count": len(data),
            "column_count": len(data.columns),
            "memory_usage": data.memory_usage(deep=True).sum(),
            "null_counts": data.isnull().sum().to_dict(),
            "dtypes": data.dtypes.astype(str).to_dict()
        }
        
        # Check for empty dataframe
        if data.empty:
            validation_results["valid"] = False
            validation_results["errors"].append("Dataset is empty")
            return validation_results
        
        # Check for duplicate columns
        if len(data.columns) != len(set(data.columns)):
            validation_results["warnings"].append("Duplicate column names detected")
        
        # Check for high null percentages
        null_percentages = (data.isnull().sum() / len(data)) * 100
        high_null_cols = null_percentages[null_percentages > 50].to_dict()
        if high_null_cols:
            validation_results["warnings"].append(
                f"High null percentage in columns: {high_null_cols}"
            )
        
        # LLM-enhanced validation if enabled
        if self.config.llm_enhanced and self.llm_service:
            llm_validation = await self._llm_validate_data(data, schema)
            validation_results["llm_insights"] = llm_validation
        
        return validation_results
    
    async def _llm_validate_data(
        self, 
        data: pd.DataFrame,
        schema: Optional[DataSchema] = None
    ) -> Dict[str, Any]:
        """Use LLM for advanced data validation"""
        # Sample data for LLM analysis
        sample = data.head(self.config.sample_size).to_dict('records')
        
        # Prepare statistics
        stats = {
            "shape": data.shape,
            "dtypes": data.dtypes.astype(str).to_dict(),
            "null_counts": data.isnull().sum().to_dict(),
            "unique_counts": {col: data[col].nunique() for col in data.columns},
            "sample_data": sample[:10]  # First 10 rows
        }
        
        # Get quality assessment from LLM
        quality_assessment = await self.llm_service.assess_data_quality(stats)
        
        return quality_assessment
    
    async def transform_data(
        self,
        data: pd.DataFrame,
        transformations: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Apply transformations to data"""
        transformed = data.copy()
        
        for transform in transformations:
            transform_type = transform.get("type")
            
            if transform_type == "rename":
                transformed = transformed.rename(columns=transform["mapping"])
            
            elif transform_type == "cast":
                for col, dtype in transform["types"].items():
                    if col in transformed.columns:
                        try:
                            if dtype == "datetime":
                                transformed[col] = pd.to_datetime(transformed[col])
                            else:
                                transformed[col] = transformed[col].astype(dtype)
                        except Exception as e:
                            logger.warning(f"Failed to cast {col} to {dtype}: {str(e)}")
            
            elif transform_type == "filter":
                transformed = transformed.query(transform["condition"])
            
            elif transform_type == "llm":
                # LLM-based transformation
                if self.config.llm_enhanced and self.llm_service:
                    transformed = await self._llm_transform(transformed, transform)
        
        return transformed
    
    async def _llm_transform(
        self,
        data: pd.DataFrame,
        transform_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply LLM-based transformation"""
        # This would be implemented based on specific transformation needs
        # For example: entity extraction, text normalization, etc.
        return data
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return []
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema for this connector"""
        return {
            "type": "object",
            "properties": {
                "llm_enhanced": {
                    "type": "boolean",
                    "description": "Enable LLM-enhanced processing"
                },
                "auto_schema": {
                    "type": "boolean",
                    "description": "Automatically infer schema"
                },
                "quality_check": {
                    "type": "boolean",
                    "description": "Perform quality checks"
                }
            }
        }