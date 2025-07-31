from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import re
from collections import Counter

from services.llm.llm_factory import LLMFactory
from core.logging import get_logger

logger = get_logger(__name__)


class QualityIssueType(str, Enum):
    """Types of data quality issues"""
    MISSING_VALUES = "missing_values"
    DUPLICATES = "duplicates"
    OUTLIERS = "outliers"
    INCONSISTENT_FORMATS = "inconsistent_formats"
    INVALID_VALUES = "invalid_values"
    DATA_TYPE_MISMATCH = "data_type_mismatch"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    BUSINESS_RULE_VIOLATION = "business_rule_violation"
    PII_DETECTED = "pii_detected"
    ENCODING_ISSUES = "encoding_issues"


class QualitySeverity(str, Enum):
    """Severity levels for quality issues"""
    CRITICAL = "critical"  # Data unusable
    HIGH = "high"        # Major issues
    MEDIUM = "medium"    # Moderate issues
    LOW = "low"          # Minor issues
    INFO = "info"        # Informational


@dataclass
class QualityIssue:
    """Represents a data quality issue"""
    issue_type: QualityIssueType
    severity: QualitySeverity
    column: Optional[str]
    row_indices: Optional[List[int]]
    description: str
    impact: str
    recommendation: str
    sample_data: Optional[Any] = None
    metrics: Optional[Dict[str, Any]] = None


@dataclass
class QualityReport:
    """Comprehensive quality assessment report"""
    overall_score: float  # 0-100
    total_issues: int
    critical_issues: int
    issues: List[QualityIssue]
    column_scores: Dict[str, float]
    data_profile: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime
    processing_time: float


class QualityValidator:
    """
    Automated data quality validation system with LLM enhancement
    """
    
    def __init__(self, llm_provider: str = "claude"):
        self.llm_service = LLMFactory.create(llm_provider)
        
        # Quality thresholds
        self.thresholds = {
            "missing_threshold": 0.1,      # 10% missing values
            "duplicate_threshold": 0.05,    # 5% duplicates
            "outlier_std_dev": 3,          # 3 standard deviations
            "min_unique_ratio": 0.01,       # 1% unique values for categorical
            "max_cardinality": 0.95        # 95% cardinality for identifiers
        }
        
        # PII patterns
        self.pii_patterns = {
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
            "ip_address": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"
        }
    
    async def validate(
        self,
        data: pd.DataFrame,
        schema: Optional[Dict[str, Any]] = None,
        business_rules: Optional[List[Dict[str, Any]]] = None,
        deep_analysis: bool = True
    ) -> QualityReport:
        """
        Perform comprehensive quality validation
        
        Args:
            data: DataFrame to validate
            schema: Expected schema definition
            business_rules: Custom business rules to validate
            deep_analysis: Whether to use LLM for deep analysis
            
        Returns:
            Comprehensive quality report
        """
        logger.info(f"Starting quality validation for dataset with shape {data.shape}")
        start_time = datetime.now()
        
        issues = []
        
        # Basic statistical profiling
        data_profile = self._profile_data(data)
        
        # Check for missing values
        missing_issues = self._check_missing_values(data)
        issues.extend(missing_issues)
        
        # Check for duplicates
        duplicate_issues = self._check_duplicates(data)
        issues.extend(duplicate_issues)
        
        # Check for outliers
        outlier_issues = self._check_outliers(data)
        issues.extend(outlier_issues)
        
        # Check data types and formats
        format_issues = self._check_formats(data, schema)
        issues.extend(format_issues)
        
        # Check for PII
        pii_issues = self._check_pii(data)
        issues.extend(pii_issues)
        
        # Check business rules
        if business_rules:
            rule_issues = await self._check_business_rules(data, business_rules)
            issues.extend(rule_issues)
        
        # Deep analysis with LLM
        if deep_analysis and len(data) > 0:
            llm_issues = await self._llm_deep_analysis(data, issues)
            issues.extend(llm_issues)
        
        # Calculate scores
        overall_score, column_scores = self._calculate_quality_scores(data, issues)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(issues, data_profile)
        
        # Count critical issues
        critical_count = sum(1 for issue in issues if issue.severity == QualitySeverity.CRITICAL)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QualityReport(
            overall_score=overall_score,
            total_issues=len(issues),
            critical_issues=critical_count,
            issues=issues,
            column_scores=column_scores,
            data_profile=data_profile,
            recommendations=recommendations,
            timestamp=datetime.now(),
            processing_time=processing_time
        )
    
    def _profile_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic data profile"""
        profile = {
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "memory_usage": data.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
            "column_stats": {}
        }
        
        for col in data.columns:
            col_data = data[col]
            stats = {
                "dtype": str(col_data.dtype),
                "null_count": col_data.isnull().sum(),
                "null_percentage": (col_data.isnull().sum() / len(data)) * 100,
                "unique_count": col_data.nunique(),
                "unique_percentage": (col_data.nunique() / len(data)) * 100
            }
            
            # Numeric statistics
            if pd.api.types.is_numeric_dtype(col_data):
                stats.update({
                    "mean": col_data.mean(),
                    "std": col_data.std(),
                    "min": col_data.min(),
                    "max": col_data.max(),
                    "q25": col_data.quantile(0.25),
                    "q50": col_data.quantile(0.50),
                    "q75": col_data.quantile(0.75)
                })
            
            # Categorical statistics
            elif pd.api.types.is_object_dtype(col_data):
                value_counts = col_data.value_counts()
                stats.update({
                    "top_values": value_counts.head(5).to_dict(),
                    "mode": value_counts.index[0] if len(value_counts) > 0 else None
                })
            
            profile["column_stats"][col] = stats
        
        return profile
    
    def _check_missing_values(self, data: pd.DataFrame) -> List[QualityIssue]:
        """Check for missing values"""
        issues = []
        
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            missing_ratio = missing_count / len(data)
            
            if missing_count > 0:
                severity = self._determine_missing_severity(missing_ratio)
                
                # Get indices of missing values (limit to first 100)
                missing_indices = data[data[col].isnull()].index.tolist()[:100]
                
                issue = QualityIssue(
                    issue_type=QualityIssueType.MISSING_VALUES,
                    severity=severity,
                    column=col,
                    row_indices=missing_indices,
                    description=f"Column '{col}' has {missing_count} missing values ({missing_ratio:.1%})",
                    impact=self._get_missing_impact(missing_ratio),
                    recommendation=self._get_missing_recommendation(col, missing_ratio),
                    metrics={"count": missing_count, "ratio": missing_ratio}
                )
                issues.append(issue)
        
        return issues
    
    def _check_duplicates(self, data: pd.DataFrame) -> List[QualityIssue]:
        """Check for duplicate rows"""
        issues = []
        
        # Full row duplicates
        duplicates = data.duplicated()
        dup_count = duplicates.sum()
        
        if dup_count > 0:
            dup_ratio = dup_count / len(data)
            dup_indices = data[duplicates].index.tolist()[:100]
            
            issue = QualityIssue(
                issue_type=QualityIssueType.DUPLICATES,
                severity=QualitySeverity.HIGH if dup_ratio > self.thresholds["duplicate_threshold"] else QualitySeverity.MEDIUM,
                column=None,
                row_indices=dup_indices,
                description=f"Found {dup_count} duplicate rows ({dup_ratio:.1%})",
                impact="Duplicate data can skew analysis results and waste storage",
                recommendation="Remove duplicates using df.drop_duplicates() or investigate source",
                metrics={"count": dup_count, "ratio": dup_ratio}
            )
            issues.append(issue)
        
        # Check for duplicate IDs in columns that look like identifiers
        for col in data.columns:
            if self._is_likely_identifier(col, data[col]):
                col_dups = data[col].duplicated()
                col_dup_count = col_dups.sum()
                
                if col_dup_count > 0:
                    issue = QualityIssue(
                        issue_type=QualityIssueType.DUPLICATES,
                        severity=QualitySeverity.CRITICAL,
                        column=col,
                        row_indices=data[col_dups].index.tolist()[:100],
                        description=f"Identifier column '{col}' has {col_dup_count} duplicate values",
                        impact="Duplicate identifiers violate data integrity",
                        recommendation=f"Investigate duplicate {col} values - these should be unique",
                        metrics={"count": col_dup_count}
                    )
                    issues.append(issue)
        
        return issues
    
    def _check_outliers(self, data: pd.DataFrame) -> List[QualityIssue]:
        """Check for statistical outliers in numeric columns"""
        issues = []
        
        for col in data.select_dtypes(include=[np.number]).columns:
            col_data = data[col].dropna()
            
            if len(col_data) < 10:  # Skip if too few values
                continue
            
            # Z-score method
            mean = col_data.mean()
            std = col_data.std()
            
            if std > 0:
                z_scores = np.abs((col_data - mean) / std)
                outliers = z_scores > self.thresholds["outlier_std_dev"]
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    outlier_indices = col_data[outliers].index.tolist()[:50]
                    outlier_values = col_data[outliers].tolist()[:10]
                    
                    issue = QualityIssue(
                        issue_type=QualityIssueType.OUTLIERS,
                        severity=QualitySeverity.MEDIUM,
                        column=col,
                        row_indices=outlier_indices,
                        description=f"Column '{col}' has {outlier_count} outliers (>{self.thresholds['outlier_std_dev']} std dev)",
                        impact="Outliers can distort statistical analyses and model training",
                        recommendation="Review outliers - they may be errors or legitimate extreme values",
                        sample_data=outlier_values,
                        metrics={
                            "count": outlier_count,
                            "mean": mean,
                            "std": std,
                            "threshold": self.thresholds["outlier_std_dev"]
                        }
                    )
                    issues.append(issue)
            
            # IQR method for additional validation
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            
            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                iqr_outliers = (col_data < lower_bound) | (col_data > upper_bound)
                iqr_outlier_count = iqr_outliers.sum()
                
                # Only report if significantly different from z-score method
                if iqr_outlier_count > outlier_count * 1.5:
                    issue = QualityIssue(
                        issue_type=QualityIssueType.OUTLIERS,
                        severity=QualitySeverity.LOW,
                        column=col,
                        row_indices=col_data[iqr_outliers].index.tolist()[:50],
                        description=f"Column '{col}' has {iqr_outlier_count} IQR outliers",
                        impact="Additional outliers detected using IQR method",
                        recommendation="Consider domain knowledge to validate these values",
                        metrics={
                            "count": iqr_outlier_count,
                            "lower_bound": lower_bound,
                            "upper_bound": upper_bound
                        }
                    )
                    issues.append(issue)
        
        return issues
    
    def _check_formats(self, data: pd.DataFrame, schema: Optional[Dict[str, Any]]) -> List[QualityIssue]:
        """Check data formats and consistency"""
        issues = []
        
        for col in data.columns:
            col_data = data[col]
            
            # Skip if all null
            if col_data.isnull().all():
                continue
            
            # Check date formats
            if self._looks_like_date(col, col_data):
                date_issues = self._check_date_format(col, col_data)
                issues.extend(date_issues)
            
            # Check numeric formats
            elif pd.api.types.is_object_dtype(col_data) and self._looks_like_numeric(col_data):
                numeric_issues = self._check_numeric_format(col, col_data)
                issues.extend(numeric_issues)
            
            # Check string consistency
            elif pd.api.types.is_string_dtype(col_data):
                string_issues = self._check_string_consistency(col, col_data)
                issues.extend(string_issues)
        
        # Check against schema if provided
        if schema:
            schema_issues = self._check_schema_compliance(data, schema)
            issues.extend(schema_issues)
        
        return issues
    
    def _check_pii(self, data: pd.DataFrame) -> List[QualityIssue]:
        """Check for potential PII data"""
        issues = []
        
        for col in data.select_dtypes(include=[object]).columns:
            col_data = data[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            # Sample for efficiency
            sample_size = min(1000, len(col_data))
            sample_data = col_data.sample(n=sample_size)
            
            for pii_type, pattern in self.pii_patterns.items():
                matches = sample_data.astype(str).str.contains(pattern, regex=True, na=False)
                match_count = matches.sum()
                
                if match_count > 0:
                    # Estimate total count
                    estimated_total = int(match_count * len(col_data) / sample_size)
                    
                    issue = QualityIssue(
                        issue_type=QualityIssueType.PII_DETECTED,
                        severity=QualitySeverity.CRITICAL,
                        column=col,
                        row_indices=sample_data[matches].index.tolist()[:20],
                        description=f"Column '{col}' contains potential {pii_type.upper()} data",
                        impact="PII exposure poses privacy and compliance risks",
                        recommendation=f"Mask or remove {pii_type} data before processing",
                        metrics={
                            "detected_count": match_count,
                            "estimated_total": estimated_total,
                            "pii_type": pii_type
                        }
                    )
                    issues.append(issue)
        
        return issues
    
    async def _check_business_rules(
        self,
        data: pd.DataFrame,
        rules: List[Dict[str, Any]]
    ) -> List[QualityIssue]:
        """Check custom business rules"""
        issues = []
        
        for rule in rules:
            rule_type = rule.get("type")
            
            if rule_type == "range":
                col = rule["column"]
                min_val = rule.get("min")
                max_val = rule.get("max")
                
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                    violations = 0
                    violation_indices = []
                    
                    if min_val is not None:
                        below = data[col] < min_val
                        violations += below.sum()
                        violation_indices.extend(data[below].index.tolist())
                    
                    if max_val is not None:
                        above = data[col] > max_val
                        violations += above.sum()
                        violation_indices.extend(data[above].index.tolist())
                    
                    if violations > 0:
                        issue = QualityIssue(
                            issue_type=QualityIssueType.BUSINESS_RULE_VIOLATION,
                            severity=QualitySeverity.HIGH,
                            column=col,
                            row_indices=violation_indices[:100],
                            description=f"Column '{col}' violates range rule [{min_val}, {max_val}]",
                            impact="Business rule violations may indicate data errors",
                            recommendation=f"Investigate values outside the range [{min_val}, {max_val}]",
                            metrics={"violation_count": violations}
                        )
                        issues.append(issue)
            
            elif rule_type == "regex":
                col = rule["column"]
                pattern = rule["pattern"]
                
                if col in data.columns:
                    matches = data[col].astype(str).str.match(pattern, na=False)
                    violations = (~matches).sum()
                    
                    if violations > 0:
                        issue = QualityIssue(
                            issue_type=QualityIssueType.BUSINESS_RULE_VIOLATION,
                            severity=QualitySeverity.HIGH,
                            column=col,
                            row_indices=data[~matches].index.tolist()[:100],
                            description=f"Column '{col}' violates regex pattern rule",
                            impact="Format violations may cause downstream processing errors",
                            recommendation=f"Ensure all values in '{col}' match the required pattern",
                            metrics={"violation_count": violations}
                        )
                        issues.append(issue)
        
        return issues
    
    async def _llm_deep_analysis(
        self,
        data: pd.DataFrame,
        existing_issues: List[QualityIssue]
    ) -> List[QualityIssue]:
        """Use LLM for deep semantic analysis"""
        issues = []
        
        # Prepare summary for LLM
        data_summary = {
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "sample": data.head(10).to_dict() if len(data) > 0 else {},
            "existing_issues": [
                {
                    "type": issue.issue_type,
                    "column": issue.column,
                    "description": issue.description
                }
                for issue in existing_issues[:10]
            ]
        }
        
        prompt = f"""Analyze this dataset for additional quality issues not detected by standard rules:

{data_summary}

Look for:
1. Semantic inconsistencies
2. Unusual patterns or anomalies
3. Potential data entry errors
4. Column relationships that seem wrong
5. Data that doesn't match column names

Return findings as JSON array of issues."""
        
        try:
            result = await self.llm_service.complete(prompt)
            
            # Parse LLM response
            import json
            llm_findings = json.loads(result.content)
            
            for finding in llm_findings:
                issue = QualityIssue(
                    issue_type=QualityIssueType.INVALID_VALUES,
                    severity=QualitySeverity.MEDIUM,
                    column=finding.get("column"),
                    row_indices=None,
                    description=finding.get("description", "LLM-detected issue"),
                    impact=finding.get("impact", "May affect data quality"),
                    recommendation=finding.get("recommendation", "Review and validate"),
                    metrics={"source": "llm_analysis"}
                )
                issues.append(issue)
        
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
        
        return issues
    
    def _calculate_quality_scores(
        self,
        data: pd.DataFrame,
        issues: List[QualityIssue]
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate overall and per-column quality scores"""
        # Weight by severity
        severity_weights = {
            QualitySeverity.CRITICAL: 10,
            QualitySeverity.HIGH: 5,
            QualitySeverity.MEDIUM: 2,
            QualitySeverity.LOW: 1,
            QualitySeverity.INFO: 0.5
        }
        
        # Calculate column scores
        column_scores = {}
        column_issues = {}
        
        for col in data.columns:
            column_issues[col] = [
                issue for issue in issues
                if issue.column == col
            ]
        
        for col, col_issues in column_issues.items():
            if not col_issues:
                column_scores[col] = 100.0
            else:
                total_weight = sum(
                    severity_weights.get(issue.severity, 1)
                    for issue in col_issues
                )
                # Score decreases with issue weight
                column_scores[col] = max(0, 100 - (total_weight * 5))
        
        # Calculate overall score
        total_weight = sum(
            severity_weights.get(issue.severity, 1)
            for issue in issues
        )
        
        # Base score
        base_score = 100.0
        
        # Deduct points based on issues
        deduction = min(total_weight * 2, 100)  # Cap at 100
        overall_score = max(0, base_score - deduction)
        
        # Adjust for data completeness
        completeness = (data.notna().sum().sum()) / (data.shape[0] * data.shape[1])
        overall_score = overall_score * 0.8 + (completeness * 20)
        
        return overall_score, column_scores
    
    async def _generate_recommendations(self, issues: List[QualityIssue], profile: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Group issues by type
        issue_types = Counter(issue.issue_type for issue in issues)
        
        # Priority recommendations based on issue types
        if issue_types.get(QualityIssueType.PII_DETECTED, 0) > 0:
            recommendations.append(
                "CRITICAL: Remove or mask PII data immediately to ensure compliance"
            )
        
        if issue_types.get(QualityIssueType.DUPLICATES, 0) > 0:
            recommendations.append(
                "Remove duplicate rows to improve data quality and reduce storage"
            )
        
        if issue_types.get(QualityIssueType.MISSING_VALUES, 0) > 0:
            missing_cols = [issue.column for issue in issues if issue.issue_type == QualityIssueType.MISSING_VALUES]
            recommendations.append(
                f"Handle missing values in columns: {', '.join(missing_cols[:5])}. Consider imputation or removal."
            )
        
        if issue_types.get(QualityIssueType.OUTLIERS, 0) > 0:
            recommendations.append(
                "Review outliers to determine if they are errors or valid extreme values"
            )
        
        # LLM-powered recommendations
        if len(issues) > 5:  # Only for complex cases
            prompt = f"""Based on these data quality issues, provide 3 specific, actionable recommendations:

Issues summary:
{issue_types}

Top issues:
{[issue.description for issue in issues[:5]]}

Provide recommendations that are practical and specific."""
            
            try:
                result = await self.llm_service.complete(prompt)
                llm_recommendations = result.content.split('\n')
                recommendations.extend([r.strip() for r in llm_recommendations if r.strip()][:3])
            except Exception as e:
                logger.error(f"Failed to generate LLM recommendations: {str(e)}")
        
        return recommendations
    
    # Helper methods
    def _determine_missing_severity(self, ratio: float) -> QualitySeverity:
        """Determine severity based on missing ratio"""
        if ratio > 0.5:
            return QualitySeverity.CRITICAL
        elif ratio > 0.2:
            return QualitySeverity.HIGH
        elif ratio > self.thresholds["missing_threshold"]:
            return QualitySeverity.MEDIUM
        else:
            return QualitySeverity.LOW
    
    def _get_missing_impact(self, ratio: float) -> str:
        """Get impact description for missing values"""
        if ratio > 0.5:
            return "Severe data loss - column may be unusable"
        elif ratio > 0.2:
            return "Significant missing data affecting analysis reliability"
        else:
            return "Missing data may bias results if not handled properly"
    
    def _get_missing_recommendation(self, column: str, ratio: float) -> str:
        """Get recommendation for missing values"""
        if ratio > 0.5:
            return f"Consider dropping column '{column}' or investigating data collection issues"
        elif ratio > 0.2:
            return f"Impute missing values in '{column}' or use techniques that handle missing data"
        else:
            return f"Handle missing values in '{column}' using appropriate imputation methods"
    
    def _is_likely_identifier(self, col_name: str, col_data: pd.Series) -> bool:
        """Check if column is likely an identifier"""
        # Name patterns
        id_patterns = ['id', 'key', 'code', 'number', 'identifier']
        if any(pattern in col_name.lower() for pattern in id_patterns):
            return True
        
        # High cardinality
        if len(col_data) > 100:
            cardinality = col_data.nunique() / len(col_data)
            if cardinality > self.thresholds["max_cardinality"]:
                return True
        
        return False
    
    def _looks_like_date(self, col_name: str, col_data: pd.Series) -> bool:
        """Check if column likely contains dates"""
        date_patterns = ['date', 'time', 'datetime', 'timestamp', 'created', 'updated', 'modified']
        return any(pattern in col_name.lower() for pattern in date_patterns)
    
    def _looks_like_numeric(self, col_data: pd.Series) -> bool:
        """Check if string column contains numeric data"""
        sample = col_data.dropna().head(100)
        if len(sample) == 0:
            return False
        
        # Try converting to numeric
        try:
            pd.to_numeric(sample, errors='coerce')
            # Check if most values are convertible
            numeric_count = pd.to_numeric(sample, errors='coerce').notna().sum()
            return numeric_count / len(sample) > 0.8
        except:
            return False
    
    def _check_date_format(self, col: str, col_data: pd.Series) -> List[QualityIssue]:
        """Check date format consistency"""
        issues = []
        sample = col_data.dropna().astype(str).head(1000)
        
        if len(sample) == 0:
            return issues
        
        # Common date formats
        date_formats = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
        ]
        
        format_counts = {}
        unmatched = 0
        
        for value in sample:
            matched = False
            for fmt in date_formats:
                if re.match(fmt, value):
                    format_counts[fmt] = format_counts.get(fmt, 0) + 1
                    matched = True
                    break
            if not matched:
                unmatched += 1
        
        # Check for inconsistent formats
        if len(format_counts) > 1 or unmatched > len(sample) * 0.1:
            issue = QualityIssue(
                issue_type=QualityIssueType.INCONSISTENT_FORMATS,
                severity=QualitySeverity.MEDIUM,
                column=col,
                row_indices=None,
                description=f"Column '{col}' has inconsistent date formats",
                impact="Inconsistent formats may cause parsing errors",
                recommendation=f"Standardize date format in column '{col}'",
                metrics={"format_counts": format_counts, "unmatched": unmatched}
            )
            issues.append(issue)
        
        return issues
    
    def _check_numeric_format(self, col: str, col_data: pd.Series) -> List[QualityIssue]:
        """Check numeric format in string columns"""
        issues = []
        
        # Try to convert
        numeric_converted = pd.to_numeric(col_data, errors='coerce')
        failed_count = numeric_converted.isna().sum() - col_data.isna().sum()
        
        if failed_count > 0:
            failed_indices = col_data[numeric_converted.isna() & col_data.notna()].index.tolist()[:50]
            
            issue = QualityIssue(
                issue_type=QualityIssueType.DATA_TYPE_MISMATCH,
                severity=QualitySeverity.HIGH,
                column=col,
                row_indices=failed_indices,
                description=f"Column '{col}' contains {failed_count} non-numeric values in numeric context",
                impact="Type mismatches will cause calculation errors",
                recommendation=f"Clean non-numeric values in column '{col}' or change data type",
                metrics={"failed_conversions": failed_count}
            )
            issues.append(issue)
        
        return issues
    
    def _check_string_consistency(self, col: str, col_data: pd.Series) -> List[QualityIssue]:
        """Check string data consistency"""
        issues = []
        
        # Check for leading/trailing spaces
        stripped = col_data.astype(str).str.strip()
        space_issues = (stripped != col_data.astype(str)).sum()
        
        if space_issues > 0:
            issue = QualityIssue(
                issue_type=QualityIssueType.INCONSISTENT_FORMATS,
                severity=QualitySeverity.LOW,
                column=col,
                row_indices=col_data[stripped != col_data.astype(str)].index.tolist()[:50],
                description=f"Column '{col}' has {space_issues} values with extra whitespace",
                impact="Whitespace can cause matching and grouping issues",
                recommendation=f"Strip whitespace from column '{col}'",
                metrics={"count": space_issues}
            )
            issues.append(issue)
        
        # Check for inconsistent casing
        if col_data.nunique() < 1000:  # Only for categorical-like columns
            lower_mapping = col_data.astype(str).str.lower().value_counts()
            
            # Find values that differ only in case
            case_issues = []
            for value in col_data.unique():
                if pd.notna(value):
                    lower_val = str(value).lower()
                    similar = [v for v in col_data.unique() 
                             if pd.notna(v) and str(v).lower() == lower_val and v != value]
                    if similar:
                        case_issues.append((value, similar))
            
            if case_issues:
                issue = QualityIssue(
                    issue_type=QualityIssueType.INCONSISTENT_FORMATS,
                    severity=QualitySeverity.MEDIUM,
                    column=col,
                    row_indices=None,
                    description=f"Column '{col}' has inconsistent casing",
                    impact="Case differences may cause incorrect grouping",
                    recommendation=f"Standardize casing in column '{col}'",
                    sample_data=case_issues[:5]
                )
                issues.append(issue)
        
        return issues
    
    def _check_schema_compliance(self, data: pd.DataFrame, schema: Dict[str, Any]) -> List[QualityIssue]:
        """Check compliance with provided schema"""
        issues = []
        
        # Check required columns
        required_cols = schema.get("required_columns", [])
        missing_required = set(required_cols) - set(data.columns)
        
        if missing_required:
            issue = QualityIssue(
                issue_type=QualityIssueType.INVALID_VALUES,
                severity=QualitySeverity.CRITICAL,
                column=None,
                row_indices=None,
                description=f"Missing required columns: {', '.join(missing_required)}",
                impact="Schema violations prevent proper data processing",
                recommendation="Add missing required columns or update schema",
                metrics={"missing_columns": list(missing_required)}
            )
            issues.append(issue)
        
        # Check data types
        expected_types = schema.get("column_types", {})
        for col, expected_type in expected_types.items():
            if col in data.columns:
                actual_type = str(data[col].dtype)
                
                # Simple type matching
                type_mismatch = False
                if expected_type == "integer" and not pd.api.types.is_integer_dtype(data[col]):
                    type_mismatch = True
                elif expected_type == "float" and not pd.api.types.is_float_dtype(data[col]):
                    type_mismatch = True
                elif expected_type == "string" and not pd.api.types.is_string_dtype(data[col]):
                    type_mismatch = True
                elif expected_type == "boolean" and not pd.api.types.is_bool_dtype(data[col]):
                    type_mismatch = True
                elif expected_type == "datetime" and not pd.api.types.is_datetime64_any_dtype(data[col]):
                    type_mismatch = True
                
                if type_mismatch:
                    issue = QualityIssue(
                        issue_type=QualityIssueType.DATA_TYPE_MISMATCH,
                        severity=QualitySeverity.HIGH,
                        column=col,
                        row_indices=None,
                        description=f"Column '{col}' type mismatch: expected {expected_type}, got {actual_type}",
                        impact="Type mismatches may cause processing errors",
                        recommendation=f"Convert column '{col}' to {expected_type} type",
                        metrics={"expected": expected_type, "actual": actual_type}
                    )
                    issues.append(issue)
        
        return issues