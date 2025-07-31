from typing import Dict, Type, Any, Optional, List
import pandas as pd

from .base import BasePipelineStep, StepConfig, StepResult, StepType, StepStatus
from connectors import IntelligentCSVConnector, IntelligentJSONConnector
from services.llm.llm_factory import LLMFactory
from core.logging import get_logger
from .steps.multimodal_steps import (
    DocumentParseStep,
    PDFTableExtractStep,
    ImageDataExtractStep,
    MultiModalFusionStep,
    DocumentQualityCheckStep
)
from .steps.quality_steps import (
    QualityValidationStep,
    DataCleaningStep,
    PIIDetectionStep,
    DataProfilingStep
)

logger = get_logger(__name__)


class ConnectorStep(BasePipelineStep):
    """Step for loading data from various sources"""
    
    async def execute(self, input_data: Any, context: Dict[str, Any]) -> StepResult:
        start_time = pd.Timestamp.now()
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            started_at=start_time
        )
        
        try:
            connector_type = self.config.config.get("connector_type", "csv")
            
            # Create appropriate connector
            if connector_type == "csv":
                connector = IntelligentCSVConnector(self.config.config)
            elif connector_type == "json":
                connector = IntelligentJSONConnector(self.config.config)
            else:
                raise ValueError(f"Unsupported connector type: {connector_type}")
            
            # Connect and read data
            connect_result = await connector.connect()
            if not connect_result.success:
                raise Exception(f"Connection failed: {connect_result.error}")
            
            read_result = await connector.read_data()
            if not read_result.success:
                raise Exception(f"Read failed: {read_result.error}")
            
            # Update result
            result.status = StepStatus.COMPLETED
            result.output_data = read_result.data
            result.output_schema = read_result.schema.dict() if read_result.schema else None
            result.row_count = len(read_result.data) if read_result.data is not None else 0
            result.warnings = read_result.warnings
            
        except Exception as e:
            logger.error(f"Connector step {self.name} failed: {str(e)}")
            result.status = StepStatus.FAILED
            result.error = str(e)
        
        finally:
            result.completed_at = pd.Timestamp.now()
            result.execution_time = (result.completed_at - start_time).total_seconds()
        
        return result
    
    def validate_config(self) -> bool:
        required = ["connector_type"]
        return all(field in self.config.config for field in required)
    
    def estimate_cost(self, input_size: int) -> Dict[str, float]:
        # Connectors mainly have compute cost, minimal LLM usage
        return {
            "compute_cost": 0.01,  # Minimal compute cost
            "llm_cost": 0.05 if self.config.config.get("llm_enhanced", True) else 0
        }


class TransformStep(BasePipelineStep):
    """Step for transforming data"""
    
    async def execute(self, input_data: Any, context: Dict[str, Any]) -> StepResult:
        start_time = pd.Timestamp.now()
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            started_at=start_time
        )
        
        try:
            if not isinstance(input_data, pd.DataFrame):
                raise ValueError("Transform step requires DataFrame input")
            
            df = input_data.copy()
            transform_type = self.config.config.get("type", "custom")
            
            if transform_type == "rename":
                # Rename columns
                mapping = self.config.config.get("mapping", {})
                df = df.rename(columns=mapping)
            
            elif transform_type == "filter":
                # Filter rows
                condition = self.config.config.get("condition")
                if condition:
                    df = df.query(condition)
            
            elif transform_type == "select":
                # Select columns
                columns = self.config.config.get("columns", [])
                df = df[columns]
            
            elif transform_type == "aggregate":
                # Aggregate data
                group_by = self.config.config.get("group_by", [])
                agg_funcs = self.config.config.get("aggregations", {})
                df = df.groupby(group_by).agg(agg_funcs).reset_index()
            
            elif transform_type == "custom":
                # Custom transformation code
                code = self.config.config.get("code", "")
                if code:
                    # Execute custom transformation
                    # Note: In production, this should be sandboxed
                    local_vars = {"df": df, "pd": pd}
                    exec(code, {}, local_vars)
                    df = local_vars.get("df", df)
            
            result.status = StepStatus.COMPLETED
            result.output_data = df
            result.row_count = len(df)
            
        except Exception as e:
            logger.error(f"Transform step {self.name} failed: {str(e)}")
            result.status = StepStatus.FAILED
            result.error = str(e)
        
        finally:
            result.completed_at = pd.Timestamp.now()
            result.execution_time = (result.completed_at - start_time).total_seconds()
        
        return result
    
    def validate_config(self) -> bool:
        return "type" in self.config.config
    
    def estimate_cost(self, input_size: int) -> Dict[str, float]:
        return {"compute_cost": 0.001 * (input_size / 1000)}  # Scale with data size


class LLMProcessStep(BasePipelineStep):
    """Step for LLM-powered data processing"""
    
    async def execute(self, input_data: Any, context: Dict[str, Any]) -> StepResult:
        start_time = pd.Timestamp.now()
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            started_at=start_time
        )
        
        try:
            # Get LLM service
            provider = self.config.llm_provider or "claude"
            model = self.config.llm_model
            llm_service = LLMFactory.create(provider, model)
            
            process_type = self.config.config.get("type", "enrich")
            
            if process_type == "enrich":
                # Enrich data with LLM insights
                df = input_data if isinstance(input_data, pd.DataFrame) else pd.DataFrame(input_data)
                
                # Sample for LLM processing
                sample_size = min(100, len(df))
                sample = df.head(sample_size)
                
                # Get enrichment suggestions
                enrichment = await llm_service.infer_schema(sample.to_dict('records'))
                
                result.output_data = df
                result.output_schema = enrichment
                
            elif process_type == "generate_code":
                # Generate transformation code
                source_schema = context.get("source_schema")
                target_schema = self.config.config.get("target_schema")
                
                if source_schema and target_schema:
                    code = await llm_service.generate_transformation_code(
                        source_schema, target_schema
                    )
                    result.output_data = {"transformation_code": code}
            
            elif process_type == "validate":
                # Validate data quality
                df = input_data if isinstance(input_data, pd.DataFrame) else pd.DataFrame(input_data)
                
                stats = {
                    "shape": df.shape,
                    "dtypes": df.dtypes.astype(str).to_dict(),
                    "null_counts": df.isnull().sum().to_dict(),
                    "sample": df.head(10).to_dict('records')
                }
                
                quality_report = await llm_service.assess_data_quality(stats)
                result.output_data = df
                result.output_schema = quality_report
            
            # Track LLM usage
            usage_stats = llm_service.get_usage_stats()
            result.llm_tokens_used = usage_stats.get("total_tokens", 0)
            result.llm_cost = usage_stats.get("total_cost", 0)
            
            result.status = StepStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"LLM process step {self.name} failed: {str(e)}")
            result.status = StepStatus.FAILED
            result.error = str(e)
        
        finally:
            result.completed_at = pd.Timestamp.now()
            result.execution_time = (result.completed_at - start_time).total_seconds()
        
        return result
    
    def validate_config(self) -> bool:
        return "type" in self.config.config
    
    def estimate_cost(self, input_size: int) -> Dict[str, float]:
        # Estimate based on input size and LLM pricing
        tokens_estimate = input_size * 10  # Rough estimate
        cost_per_token = 0.000003  # $3 per million tokens
        
        return {
            "compute_cost": 0.01,
            "llm_cost": tokens_estimate * cost_per_token
        }


class ValidateStep(BasePipelineStep):
    """Step for data validation"""
    
    async def execute(self, input_data: Any, context: Dict[str, Any]) -> StepResult:
        start_time = pd.Timestamp.now()
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            started_at=start_time
        )
        
        try:
            df = input_data if isinstance(input_data, pd.DataFrame) else pd.DataFrame(input_data)
            
            validation_rules = self.config.config.get("rules", [])
            validation_passed = True
            validation_errors = []
            
            for rule in validation_rules:
                rule_type = rule.get("type")
                
                if rule_type == "not_null":
                    columns = rule.get("columns", [])
                    for col in columns:
                        if col in df.columns:
                            null_count = df[col].isnull().sum()
                            if null_count > 0:
                                validation_errors.append(
                                    f"Column {col} has {null_count} null values"
                                )
                                validation_passed = False
                
                elif rule_type == "unique":
                    columns = rule.get("columns", [])
                    for col in columns:
                        if col in df.columns:
                            duplicates = df[col].duplicated().sum()
                            if duplicates > 0:
                                validation_errors.append(
                                    f"Column {col} has {duplicates} duplicate values"
                                )
                                validation_passed = False
                
                elif rule_type == "range":
                    column = rule.get("column")
                    min_val = rule.get("min")
                    max_val = rule.get("max")
                    
                    if column in df.columns:
                        if min_val is not None:
                            below_min = (df[column] < min_val).sum()
                            if below_min > 0:
                                validation_errors.append(
                                    f"{below_min} values in {column} below minimum {min_val}"
                                )
                                validation_passed = False
                        
                        if max_val is not None:
                            above_max = (df[column] > max_val).sum()
                            if above_max > 0:
                                validation_errors.append(
                                    f"{above_max} values in {column} above maximum {max_val}"
                                )
                                validation_passed = False
            
            if not validation_passed and self.config.config.get("fail_on_error", True):
                result.status = StepStatus.FAILED
                result.error = "; ".join(validation_errors)
            else:
                result.status = StepStatus.COMPLETED
                result.warnings = validation_errors
            
            result.output_data = df
            result.row_count = len(df)
            
        except Exception as e:
            logger.error(f"Validate step {self.name} failed: {str(e)}")
            result.status = StepStatus.FAILED
            result.error = str(e)
        
        finally:
            result.completed_at = pd.Timestamp.now()
            result.execution_time = (result.completed_at - start_time).total_seconds()
        
        return result
    
    def validate_config(self) -> bool:
        return True  # Validation rules are optional
    
    def estimate_cost(self, input_size: int) -> Dict[str, float]:
        return {"compute_cost": 0.001 * (input_size / 1000)}


class StepRegistry:
    """Registry for pipeline steps"""
    
    _steps: Dict[StepType, Type[BasePipelineStep]] = {
        StepType.CONNECTOR: ConnectorStep,
        StepType.TRANSFORM: TransformStep,
        StepType.LLM_PROCESS: LLMProcessStep,
        StepType.VALIDATE: ValidateStep,
    }
    
    # Additional custom steps not in StepType enum
    _custom_steps: Dict[str, Type[BasePipelineStep]] = {
        "document_parse": DocumentParseStep,
        "pdf_table_extract": PDFTableExtractStep,
        "image_data_extract": ImageDataExtractStep,
        "multimodal_fusion": MultiModalFusionStep,
        "document_quality_check": DocumentQualityCheckStep,
        "quality_validation": QualityValidationStep,
        "data_cleaning": DataCleaningStep,
        "pii_detection": PIIDetectionStep,
        "data_profiling": DataProfilingStep,
    }
    
    @classmethod
    def create_step(cls, config: StepConfig) -> BasePipelineStep:
        """Create a step instance from configuration"""
        # Check standard step types first
        step_class = cls._steps.get(config.type)
        
        # If not found, check custom steps by name
        if not step_class and hasattr(config, 'name'):
            step_class = cls._custom_steps.get(config.name)
        
        # Also check by type string for custom steps
        if not step_class and hasattr(config, 'type'):
            type_str = config.type.value if hasattr(config.type, 'value') else str(config.type)
            step_class = cls._custom_steps.get(type_str)
        
        if not step_class:
            raise ValueError(f"Unknown step type: {config.type}")
        
        return step_class(config.name, config)
    
    @classmethod
    def register_step(cls, step_type: StepType, step_class: Type[BasePipelineStep]):
        """Register a custom step type"""
        cls._steps[step_type] = step_class
    
    @classmethod
    def get_available_steps(cls) -> List[str]:
        """Get list of available step types"""
        standard_steps = [step_type.value for step_type in cls._steps.keys()]
        custom_steps = list(cls._custom_steps.keys())
        return standard_steps + custom_steps


# Convenience function
def PipelineStep(config: StepConfig) -> BasePipelineStep:
    """Create a pipeline step from configuration"""
    return StepRegistry.create_step(config)