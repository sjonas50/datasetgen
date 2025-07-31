import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from services.quality.quality_validator import (
    QualityValidator,
    QualityIssueType,
    QualitySeverity
)
from services.pipeline.steps.quality_steps import (
    QualityValidationStep,
    DataCleaningStep,
    PIIDetectionStep,
    DataProfilingStep
)
from services.pipeline.base import StepInput


class TestQualityValidator:
    """Test the quality validator"""
    
    @pytest.fixture
    def validator(self):
        return QualityValidator()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with various quality issues"""
        np.random.seed(42)
        
        return pd.DataFrame({
            "customer_id": ["CUST001", "CUST002", "CUST002", "CUST003", None, "CUST005"],
            "email": ["john@example.com", "jane@test.com", "jane@test.com", "invalid-email", None, "bob@example.com"],
            "phone": ["123-456-7890", "987-654-3210", "987-654-3210", "invalid", "555-555-5555", None],
            "amount": [100.50, 200.75, 200.75, -50.0, 10000.0, 150.25],
            "date": ["2024-01-01", "2024-01-02", "2024-01-02", "2024/01/03", "invalid", "2024-01-05"],
            "status": ["active", "ACTIVE", "Active", "inactive", None, "  pending  "]
        })
    
    @pytest.mark.asyncio
    async def test_missing_value_detection(self, validator, sample_data):
        """Test detection of missing values"""
        report = await validator.validate(sample_data, deep_analysis=False)
        
        missing_issues = [
            issue for issue in report.issues
            if issue.issue_type == QualityIssueType.MISSING_VALUES
        ]
        
        assert len(missing_issues) > 0
        
        # Check specific columns
        missing_cols = {issue.column for issue in missing_issues}
        assert "customer_id" in missing_cols
        assert "email" in missing_cols
        assert "phone" in missing_cols
        assert "status" in missing_cols
    
    @pytest.mark.asyncio
    async def test_duplicate_detection(self, validator, sample_data):
        """Test detection of duplicates"""
        report = await validator.validate(sample_data, deep_analysis=False)
        
        dup_issues = [
            issue for issue in report.issues
            if issue.issue_type == QualityIssueType.DUPLICATES
        ]
        
        assert len(dup_issues) > 0
        
        # Should detect duplicate rows
        full_dup = next(
            (issue for issue in dup_issues if issue.column is None),
            None
        )
        assert full_dup is not None
        assert full_dup.metrics["count"] == 1  # One duplicate row
    
    @pytest.mark.asyncio
    async def test_outlier_detection(self, validator, sample_data):
        """Test detection of outliers"""
        report = await validator.validate(sample_data, deep_analysis=False)
        
        outlier_issues = [
            issue for issue in report.issues
            if issue.issue_type == QualityIssueType.OUTLIERS
        ]
        
        # Should detect amount outliers (10000.0 and -50.0)
        amount_outliers = [
            issue for issue in outlier_issues
            if issue.column == "amount"
        ]
        assert len(amount_outliers) > 0
    
    @pytest.mark.asyncio
    async def test_pii_detection(self, validator, sample_data):
        """Test PII detection"""
        report = await validator.validate(sample_data, deep_analysis=False)
        
        pii_issues = [
            issue for issue in report.issues
            if issue.issue_type == QualityIssueType.PII_DETECTED
        ]
        
        assert len(pii_issues) > 0
        
        # Check detected PII types
        pii_types = {issue.metrics["pii_type"] for issue in pii_issues}
        assert "email" in pii_types
        assert "phone" in pii_types
    
    @pytest.mark.asyncio
    async def test_format_consistency(self, validator, sample_data):
        """Test format consistency checks"""
        report = await validator.validate(sample_data, deep_analysis=False)
        
        format_issues = [
            issue for issue in report.issues
            if issue.issue_type == QualityIssueType.INCONSISTENT_FORMATS
        ]
        
        assert len(format_issues) > 0
        
        # Should detect inconsistent date formats
        date_issues = [
            issue for issue in format_issues
            if issue.column == "date"
        ]
        assert len(date_issues) > 0
        
        # Should detect inconsistent casing in status
        status_issues = [
            issue for issue in format_issues
            if issue.column == "status"
        ]
        assert len(status_issues) > 0
    
    @pytest.mark.asyncio
    async def test_business_rules(self, validator, sample_data):
        """Test business rule validation"""
        business_rules = [
            {
                "type": "range",
                "column": "amount",
                "min": 0,
                "max": 1000
            },
            {
                "type": "regex",
                "column": "customer_id",
                "pattern": "^CUST[0-9]{3}$"
            }
        ]
        
        report = await validator.validate(
            sample_data,
            business_rules=business_rules,
            deep_analysis=False
        )
        
        rule_violations = [
            issue for issue in report.issues
            if issue.issue_type == QualityIssueType.BUSINESS_RULE_VIOLATION
        ]
        
        assert len(rule_violations) > 0
        
        # Check amount violations
        amount_violations = [
            issue for issue in rule_violations
            if issue.column == "amount"
        ]
        assert len(amount_violations) > 0
    
    @pytest.mark.asyncio
    async def test_quality_scoring(self, validator):
        """Test quality score calculation"""
        # Perfect data
        perfect_data = pd.DataFrame({
            "id": range(100),
            "value": np.random.normal(50, 10, 100)
        })
        
        report = await validator.validate(perfect_data, deep_analysis=False)
        assert report.overall_score > 90  # Should have high score
        
        # Poor quality data
        poor_data = pd.DataFrame({
            "id": [1, 1, 2, None, None],  # Duplicates and missing
            "value": [1, 1, None, None, 1000]  # Missing and outlier
        })
        
        report = await validator.validate(poor_data, deep_analysis=False)
        assert report.overall_score < 50  # Should have low score


class TestQualitySteps:
    """Test pipeline steps for quality validation"""
    
    @pytest.fixture
    def sample_input(self):
        """Create sample pipeline input"""
        data = pd.DataFrame({
            "id": [1, 2, 2, 3, 4],
            "email": ["test@example.com", "user@test.com", "user@test.com", None, "admin@example.com"],
            "value": [10, 20, 20, 30, 1000],
            "status": ["active", "ACTIVE", "Active", "inactive", "  pending  "]
        })
        
        return StepInput(data=data, config={}, metadata={})
    
    @pytest.mark.asyncio
    async def test_quality_validation_step(self, sample_input):
        """Test quality validation pipeline step"""
        step = QualityValidationStep()
        
        config = {
            "min_quality_score": 50.0,
            "fail_on_critical": False,
            "add_quality_column": True
        }
        
        input_with_config = StepInput(
            data=sample_input.data,
            config=config,
            metadata={}
        )
        
        output = await step.process(input_with_config)
        
        assert output.success
        assert "quality_score" in output.metadata
        assert "total_issues" in output.metadata
        assert "issue_summary" in output.metadata
        assert "_quality_score" in output.data.columns
    
    @pytest.mark.asyncio
    async def test_data_cleaning_step(self, sample_input):
        """Test data cleaning pipeline step"""
        step = DataCleaningStep()
        
        config = {
            "operations": [
                "remove_duplicates",
                "strip_whitespace",
                "standardize_case",
                "handle_missing"
            ],
            "case_columns": ["status"],
            "case_type": "lower",
            "missing_strategy": "fill",
            "critical_columns": ["id"]
        }
        
        input_with_config = StepInput(
            data=sample_input.data,
            config=config,
            metadata={}
        )
        
        output = await step.process(input_with_config)
        
        assert output.success
        assert len(output.data) < len(sample_input.data)  # Duplicates removed
        assert output.data["status"].str.strip().str.lower().equals(output.data["status"])  # Standardized
        assert "cleaning_operations" in output.metadata
    
    @pytest.mark.asyncio
    async def test_pii_detection_step(self, sample_input):
        """Test PII detection pipeline step"""
        step = PIIDetectionStep()
        
        config = {
            "action": "mask",
            "mask_char": "*",
            "pii_types": ["email"]
        }
        
        input_with_config = StepInput(
            data=sample_input.data,
            config=config,
            metadata={}
        )
        
        output = await step.process(input_with_config)
        
        assert output.success
        assert "pii_detected" in output.metadata
        assert output.metadata["pii_detected"]
        
        # Check if emails are masked
        for email in output.data["email"].dropna():
            assert "*" in email  # Should be masked
    
    @pytest.mark.asyncio
    async def test_data_profiling_step(self, sample_input):
        """Test data profiling pipeline step"""
        step = DataProfilingStep()
        
        config = {
            "include_correlations": True,
            "include_distributions": True
        }
        
        input_with_config = StepInput(
            data=sample_input.data,
            config=config,
            metadata={}
        )
        
        output = await step.process(input_with_config)
        
        assert output.success
        assert "profile" in output.metadata
        assert "summary" in output.metadata
        
        profile = output.metadata["profile"]
        assert "shape" in profile
        assert "column_stats" in profile
        assert "correlations" in profile
        
        summary = output.metadata["summary"]
        assert "dataset_overview" in summary
        assert "data_quality" in summary
        assert "key_insights" in summary


@pytest.mark.integration
class TestQualityPipeline:
    """Integration tests for quality validation pipeline"""
    
    @pytest.mark.asyncio
    async def test_full_quality_pipeline(self):
        """Test complete quality validation pipeline"""
        # Create test data with various issues
        test_data = pd.DataFrame({
            "customer_id": ["CUST001", "CUST002", "CUST002", None, "CUST-BAD"],
            "email": ["john@example.com", "jane@test.com", "jane@test.com", "bad-email", None],
            "ssn": ["123-45-6789", "987-65-4321", "987-65-4321", None, "invalid"],
            "amount": [100, 200, 200, -50, 10000],
            "date": ["2024-01-01", "2024-01-02", "2024-01-02", "01/03/2024", None],
            "status": ["Active", "active", "ACTIVE", "inactive", "  pending  "]
        })
        
        # Step 1: Initial validation
        validator_step = QualityValidationStep()
        input_data = StepInput(
            data=test_data,
            config={
                "min_quality_score": 0,  # Don't fail
                "fail_on_critical": False
            },
            metadata={}
        )
        
        validation_output = await validator_step.process(input_data)
        initial_score = validation_output.metadata["quality_score"]
        
        # Step 2: PII detection and masking
        pii_step = PIIDetectionStep()
        pii_input = StepInput(
            data=validation_output.data,
            config={"action": "mask"},
            metadata=validation_output.metadata
        )
        
        pii_output = await pii_step.process(pii_input)
        assert pii_output.metadata["pii_detected"]
        
        # Step 3: Clean the data
        cleaning_step = DataCleaningStep()
        cleaning_input = StepInput(
            data=pii_output.data,
            config={
                "operations": [
                    "remove_duplicates",
                    "strip_whitespace",
                    "standardize_case",
                    "handle_missing",
                    "remove_outliers"
                ],
                "case_columns": ["status"],
                "case_type": "lower"
            },
            metadata=pii_output.metadata
        )
        
        cleaning_output = await cleaning_step.process(cleaning_input)
        
        # Step 4: Final validation
        final_validation = await validator_step.process(
            StepInput(
                data=cleaning_output.data,
                config={"min_quality_score": 0, "fail_on_critical": False},
                metadata=cleaning_output.metadata
            )
        )
        
        final_score = final_validation.metadata["quality_score"]
        
        # Quality should improve after cleaning
        assert final_score > initial_score
        assert len(cleaning_output.data) < len(test_data)  # Some rows removed
        assert cleaning_output.metadata["quality_score_after"] > initial_score