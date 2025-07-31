from typing import Dict, Any, List, Optional
import pandas as pd
import json

from services.pipeline.base import PipelineStep, StepInput, StepOutput
from services.quality.quality_validator import (
    QualityValidator,
    QualityReport,
    QualitySeverity,
    QualityIssueType
)
from core.logging import get_logger

logger = get_logger(__name__)


class QualityValidationStep(PipelineStep):
    """
    Comprehensive data quality validation step
    """
    
    def __init__(self):
        super().__init__()
        self.validator = QualityValidator()
    
    @property
    def name(self) -> str:
        return "quality_validation"
    
    @property
    def description(self) -> str:
        return "Perform comprehensive data quality validation"
    
    async def process(self, input_data: StepInput) -> StepOutput:
        data = input_data.data
        config = input_data.config or {}
        
        # Ensure we have a DataFrame
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, list):
                data = pd.DataFrame(data)
            else:
                raise ValueError("Quality validation requires tabular data")
        
        # Extract configuration
        schema = config.get("schema")
        business_rules = config.get("business_rules", [])
        deep_analysis = config.get("deep_analysis", True)
        fail_on_critical = config.get("fail_on_critical", True)
        min_quality_score = config.get("min_quality_score", 70.0)
        
        # Perform validation
        logger.info(f"Starting quality validation for data with shape {data.shape}")
        report = await self.validator.validate(
            data=data,
            schema=schema,
            business_rules=business_rules,
            deep_analysis=deep_analysis
        )
        
        # Check if we should fail
        should_fail = False
        failure_reasons = []
        
        if fail_on_critical and report.critical_issues > 0:
            should_fail = True
            failure_reasons.append(f"Found {report.critical_issues} critical issues")
        
        if report.overall_score < min_quality_score:
            should_fail = True
            failure_reasons.append(
                f"Quality score {report.overall_score:.1f} below minimum {min_quality_score}"
            )
        
        # Prepare output
        output_data = data  # Pass through original data
        
        # Add quality report to metadata
        metadata = {
            "quality_score": report.overall_score,
            "total_issues": report.total_issues,
            "critical_issues": report.critical_issues,
            "issue_summary": self._summarize_issues(report),
            "recommendations": report.recommendations[:5],
            "column_scores": report.column_scores,
            "validation_passed": not should_fail
        }
        
        # If configured, add quality score as column
        if config.get("add_quality_column", False):
            output_data = output_data.copy()
            output_data["_quality_score"] = report.overall_score
        
        # Create output
        output = StepOutput(
            data=output_data,
            metadata=metadata,
            success=not should_fail
        )
        
        if should_fail:
            output.error = f"Quality validation failed: {'; '.join(failure_reasons)}"
            # Add detailed report to output for debugging
            output.metadata["quality_report"] = self._serialize_report(report)
        
        return output
    
    def _summarize_issues(self, report: QualityReport) -> Dict[str, Any]:
        """Summarize issues by type and severity"""
        summary = {
            "by_type": {},
            "by_severity": {},
            "by_column": {}
        }
        
        for issue in report.issues:
            # By type
            issue_type = issue.issue_type.value
            if issue_type not in summary["by_type"]:
                summary["by_type"][issue_type] = 0
            summary["by_type"][issue_type] += 1
            
            # By severity
            severity = issue.severity.value
            if severity not in summary["by_severity"]:
                summary["by_severity"][severity] = 0
            summary["by_severity"][severity] += 1
            
            # By column
            if issue.column:
                if issue.column not in summary["by_column"]:
                    summary["by_column"][issue.column] = 0
                summary["by_column"][issue.column] += 1
        
        return summary
    
    def _serialize_report(self, report: QualityReport) -> Dict[str, Any]:
        """Serialize quality report for storage"""
        return {
            "overall_score": report.overall_score,
            "total_issues": report.total_issues,
            "critical_issues": report.critical_issues,
            "column_scores": report.column_scores,
            "recommendations": report.recommendations,
            "timestamp": report.timestamp.isoformat(),
            "processing_time": report.processing_time,
            "issues": [
                {
                    "type": issue.issue_type.value,
                    "severity": issue.severity.value,
                    "column": issue.column,
                    "description": issue.description,
                    "impact": issue.impact,
                    "recommendation": issue.recommendation,
                    "metrics": issue.metrics
                }
                for issue in report.issues[:100]  # Limit to top 100 issues
            ]
        }


class DataCleaningStep(PipelineStep):
    """
    Automated data cleaning based on quality issues
    """
    
    def __init__(self):
        super().__init__()
        self.validator = QualityValidator()
    
    @property
    def name(self) -> str:
        return "data_cleaning"
    
    @property
    def description(self) -> str:
        return "Automatically clean data based on quality issues"
    
    async def process(self, input_data: StepInput) -> StepOutput:
        data = input_data.data
        config = input_data.config or {}
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data cleaning requires DataFrame input")
        
        # Get cleaning configuration
        operations = config.get("operations", [
            "remove_duplicates",
            "strip_whitespace",
            "standardize_case",
            "handle_missing",
            "fix_types"
        ])
        
        cleaned_data = data.copy()
        cleaning_log = []
        
        # Remove duplicates
        if "remove_duplicates" in operations:
            before_rows = len(cleaned_data)
            cleaned_data = cleaned_data.drop_duplicates()
            removed = before_rows - len(cleaned_data)
            if removed > 0:
                cleaning_log.append(f"Removed {removed} duplicate rows")
        
        # Strip whitespace
        if "strip_whitespace" in operations:
            string_cols = cleaned_data.select_dtypes(include=[object]).columns
            for col in string_cols:
                cleaned_data[col] = cleaned_data[col].astype(str).str.strip()
            if len(string_cols) > 0:
                cleaning_log.append(f"Stripped whitespace from {len(string_cols)} columns")
        
        # Standardize case
        if "standardize_case" in operations:
            case_columns = config.get("case_columns", [])
            case_type = config.get("case_type", "lower")  # lower, upper, title
            
            for col in case_columns:
                if col in cleaned_data.columns:
                    if case_type == "lower":
                        cleaned_data[col] = cleaned_data[col].astype(str).str.lower()
                    elif case_type == "upper":
                        cleaned_data[col] = cleaned_data[col].astype(str).str.upper()
                    elif case_type == "title":
                        cleaned_data[col] = cleaned_data[col].astype(str).str.title()
            
            if case_columns:
                cleaning_log.append(f"Standardized case for {len(case_columns)} columns")
        
        # Handle missing values
        if "handle_missing" in operations:
            missing_strategy = config.get("missing_strategy", "drop")
            missing_threshold = config.get("missing_threshold", 0.5)
            
            if missing_strategy == "drop":
                # Drop columns with too many missing values
                for col in cleaned_data.columns:
                    missing_ratio = cleaned_data[col].isnull().sum() / len(cleaned_data)
                    if missing_ratio > missing_threshold:
                        cleaned_data = cleaned_data.drop(columns=[col])
                        cleaning_log.append(f"Dropped column '{col}' ({missing_ratio:.1%} missing)")
                
                # Drop rows with any missing values in critical columns
                critical_columns = config.get("critical_columns", [])
                if critical_columns:
                    before_rows = len(cleaned_data)
                    cleaned_data = cleaned_data.dropna(subset=critical_columns)
                    removed = before_rows - len(cleaned_data)
                    if removed > 0:
                        cleaning_log.append(f"Removed {removed} rows with missing critical values")
            
            elif missing_strategy == "fill":
                fill_values = config.get("fill_values", {})
                
                for col in cleaned_data.columns:
                    if col in fill_values:
                        cleaned_data[col] = cleaned_data[col].fillna(fill_values[col])
                    elif pd.api.types.is_numeric_dtype(cleaned_data[col]):
                        # Fill numeric with median
                        cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())
                    else:
                        # Fill categorical with mode or 'Unknown'
                        mode = cleaned_data[col].mode()
                        if len(mode) > 0:
                            cleaned_data[col] = cleaned_data[col].fillna(mode[0])
                        else:
                            cleaned_data[col] = cleaned_data[col].fillna("Unknown")
                
                cleaning_log.append("Filled missing values using smart imputation")
        
        # Fix data types
        if "fix_types" in operations:
            type_conversions = config.get("type_conversions", {})
            
            for col, target_type in type_conversions.items():
                if col in cleaned_data.columns:
                    try:
                        if target_type == "numeric":
                            cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
                        elif target_type == "datetime":
                            cleaned_data[col] = pd.to_datetime(cleaned_data[col], errors='coerce')
                        elif target_type == "string":
                            cleaned_data[col] = cleaned_data[col].astype(str)
                        elif target_type == "category":
                            cleaned_data[col] = cleaned_data[col].astype('category')
                        
                        cleaning_log.append(f"Converted column '{col}' to {target_type}")
                    except Exception as e:
                        logger.warning(f"Failed to convert column '{col}': {str(e)}")
        
        # Remove outliers if configured
        if "remove_outliers" in operations:
            outlier_method = config.get("outlier_method", "zscore")
            outlier_threshold = config.get("outlier_threshold", 3)
            
            numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
            total_removed = 0
            
            for col in numeric_cols:
                if outlier_method == "zscore":
                    z_scores = np.abs((cleaned_data[col] - cleaned_data[col].mean()) / cleaned_data[col].std())
                    outliers = z_scores > outlier_threshold
                    total_removed += outliers.sum()
                    cleaned_data = cleaned_data[~outliers]
            
            if total_removed > 0:
                cleaning_log.append(f"Removed {total_removed} outlier rows")
        
        # Final validation
        final_report = await self.validator.validate(
            cleaned_data,
            deep_analysis=False  # Quick validation
        )
        
        return StepOutput(
            data=cleaned_data,
            metadata={
                "rows_before": len(data),
                "rows_after": len(cleaned_data),
                "columns_before": len(data.columns),
                "columns_after": len(cleaned_data.columns),
                "cleaning_operations": cleaning_log,
                "quality_score_before": input_data.metadata.get("quality_score"),
                "quality_score_after": final_report.overall_score
            }
        )


class PIIDetectionStep(PipelineStep):
    """
    Detect and handle personally identifiable information
    """
    
    def __init__(self):
        super().__init__()
        self.validator = QualityValidator()
    
    @property
    def name(self) -> str:
        return "pii_detection"
    
    @property
    def description(self) -> str:
        return "Detect and mask PII data"
    
    async def process(self, input_data: StepInput) -> StepOutput:
        data = input_data.data
        config = input_data.config or {}
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("PII detection requires DataFrame input")
        
        # Configuration
        action = config.get("action", "mask")  # mask, remove, flag
        mask_char = config.get("mask_char", "*")
        pii_types = config.get("pii_types", ["ssn", "phone", "email", "credit_card"])
        
        # Run validation focusing on PII
        report = await self.validator.validate(data, deep_analysis=False)
        
        # Extract PII issues
        pii_issues = [
            issue for issue in report.issues
            if issue.issue_type == QualityIssueType.PII_DETECTED
        ]
        
        processed_data = data.copy()
        pii_summary = {}
        
        for issue in pii_issues:
            col = issue.column
            pii_type = issue.metrics.get("pii_type", "unknown")
            
            if pii_type not in pii_types:
                continue
            
            if col not in pii_summary:
                pii_summary[col] = []
            pii_summary[col].append(pii_type)
            
            if action == "mask":
                # Mask PII data
                pattern = self.validator.pii_patterns.get(pii_type)
                if pattern:
                    def mask_value(value):
                        if pd.isna(value):
                            return value
                        str_value = str(value)
                        # Keep first and last char for context
                        if len(str_value) > 2:
                            return str_value[0] + mask_char * (len(str_value) - 2) + str_value[-1]
                        else:
                            return mask_char * len(str_value)
                    
                    mask_indices = processed_data[col].astype(str).str.contains(
                        pattern, regex=True, na=False
                    )
                    processed_data.loc[mask_indices, col] = \
                        processed_data.loc[mask_indices, col].apply(mask_value)
            
            elif action == "remove":
                # Remove entire column
                processed_data = processed_data.drop(columns=[col])
            
            elif action == "flag":
                # Add flag column
                flag_col = f"{col}_has_pii"
                pattern = self.validator.pii_patterns.get(pii_type)
                if pattern:
                    processed_data[flag_col] = processed_data[col].astype(str).str.contains(
                        pattern, regex=True, na=False
                    )
        
        return StepOutput(
            data=processed_data,
            metadata={
                "pii_detected": len(pii_summary) > 0,
                "pii_columns": pii_summary,
                "pii_action": action,
                "total_pii_issues": len(pii_issues)
            }
        )


class DataProfilingStep(PipelineStep):
    """
    Generate comprehensive data profile
    """
    
    def __init__(self):
        super().__init__()
        self.validator = QualityValidator()
    
    @property
    def name(self) -> str:
        return "data_profiling"
    
    @property
    def description(self) -> str:
        return "Generate comprehensive data profile and statistics"
    
    async def process(self, input_data: StepInput) -> StepOutput:
        data = input_data.data
        config = input_data.config or {}
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data profiling requires DataFrame input")
        
        # Generate profile
        profile = self.validator._profile_data(data)
        
        # Add additional profiling if requested
        if config.get("include_correlations", True):
            # Calculate correlations for numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                profile["correlations"] = data[numeric_cols].corr().to_dict()
        
        if config.get("include_distributions", True):
            # Get distribution info
            profile["distributions"] = {}
            for col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    profile["distributions"][col] = {
                        "skew": data[col].skew(),
                        "kurtosis": data[col].kurtosis(),
                        "histogram": data[col].value_counts(bins=10).to_dict()
                    }
        
        # Generate summary report
        summary_report = self._generate_summary_report(profile, data)
        
        return StepOutput(
            data=data,  # Pass through original data
            metadata={
                "profile": profile,
                "summary": summary_report
            }
        )
    
    def _generate_summary_report(self, profile: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Generate human-readable summary"""
        return {
            "dataset_overview": {
                "rows": profile["shape"][0],
                "columns": profile["shape"][1],
                "memory_usage_mb": profile["memory_usage"],
                "column_types": pd.Series(profile["dtypes"]).value_counts().to_dict()
            },
            "data_quality": {
                "completeness": 1 - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])),
                "duplicate_rows": data.duplicated().sum(),
                "unique_identifiers": [
                    col for col in data.columns
                    if data[col].nunique() == len(data)
                ]
            },
            "key_insights": self._extract_key_insights(profile, data)
        }
    
    def _extract_key_insights(self, profile: Dict[str, Any], data: pd.DataFrame) -> List[str]:
        """Extract key insights from profile"""
        insights = []
        
        # Missing data insights
        high_missing = [
            col for col, stats in profile["column_stats"].items()
            if stats["null_percentage"] > 20
        ]
        if high_missing:
            insights.append(f"High missing data in columns: {', '.join(high_missing[:3])}")
        
        # Cardinality insights
        high_cardinality = [
            col for col, stats in profile["column_stats"].items()
            if stats["unique_percentage"] > 95 and stats["dtype"] == "object"
        ]
        if high_cardinality:
            insights.append(f"Potential ID columns: {', '.join(high_cardinality[:3])}")
        
        # Constant columns
        constant_cols = [
            col for col, stats in profile["column_stats"].items()
            if stats["unique_count"] == 1
        ]
        if constant_cols:
            insights.append(f"Constant columns that could be removed: {', '.join(constant_cols)}")
        
        return insights


# Export all quality steps
__all__ = [
    "QualityValidationStep",
    "DataCleaningStep",
    "PIIDetectionStep",
    "DataProfilingStep"
]