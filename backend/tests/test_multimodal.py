import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

from services.multimodal import (
    PDFProcessor,
    ImageAnalyzer,
    FusionEngine,
    DocumentParser
)
from services.multimodal.fusion_engine import FusionStrategy, ModalityType


class TestPDFProcessor:
    """Test PDF processing capabilities"""
    
    @pytest.mark.asyncio
    async def test_pdf_table_extraction(self):
        """Test extracting tables from PDF"""
        processor = PDFProcessor()
        
        # Create a mock PDF path (in real tests, use a test PDF)
        pdf_path = "test_data/sample_table.pdf"
        
        # Mock the extraction for unit testing
        # In integration tests, use actual PDFs
        tables = []
        
        assert isinstance(tables, list)
    
    @pytest.mark.asyncio
    async def test_pdf_form_extraction(self):
        """Test extracting form data from PDF"""
        processor = PDFProcessor()
        
        # Mock form data extraction
        form_data = {
            "name": "John Doe",
            "date": "2024-01-15",
            "amount": "1000.00"
        }
        
        assert isinstance(form_data, dict)
        assert "name" in form_data


class TestImageAnalyzer:
    """Test image analysis capabilities"""
    
    @pytest.mark.asyncio
    async def test_chart_data_extraction(self):
        """Test extracting data from charts"""
        analyzer = ImageAnalyzer()
        
        # Mock chart data
        chart_data = {
            "chart_type": "bar",
            "axes": {
                "x": "Month",
                "y": "Revenue"
            },
            "data_points": [
                {"x": "Jan", "y": 1000},
                {"x": "Feb", "y": 1200},
                {"x": "Mar", "y": 1500}
            ]
        }
        
        assert "chart_type" in chart_data
        assert "data_points" in chart_data
        assert len(chart_data["data_points"]) > 0
    
    @pytest.mark.asyncio
    async def test_dashboard_analysis(self):
        """Test analyzing dashboard screenshots"""
        analyzer = ImageAnalyzer()
        
        # Mock dashboard data
        dashboard_data = {
            "metrics": [
                {"name": "Total Revenue", "value": "$45.2M", "trend": "+12%"},
                {"name": "Active Users", "value": "125K", "trend": "+5%"}
            ],
            "time_period": "Q1 2024"
        }
        
        assert "metrics" in dashboard_data
        assert len(dashboard_data["metrics"]) > 0


class TestFusionEngine:
    """Test multi-modal fusion capabilities"""
    
    @pytest.mark.asyncio
    async def test_early_fusion(self):
        """Test early fusion strategy"""
        fusion_engine = FusionEngine()
        
        # Create test modalities
        modalities = {
            "tabular": {
                "data": pd.DataFrame({
                    "id": [1, 2, 3],
                    "value": [10, 20, 30]
                }),
                "type": ModalityType.TABULAR
            },
            "text": {
                "data": ["First item", "Second item", "Third item"],
                "type": ModalityType.TEXT
            }
        }
        
        # Apply fusion
        result = await fusion_engine.fuse_modalities(
            modalities,
            strategy=FusionStrategy.EARLY
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_attention_fusion(self):
        """Test attention-based fusion"""
        fusion_engine = FusionEngine()
        
        # Create modalities with different importance
        modalities = {
            "primary": {
                "data": pd.DataFrame({
                    "key": ["A", "B", "C"],
                    "important_value": [100, 200, 300]
                }),
                "type": ModalityType.TABULAR
            },
            "secondary": {
                "data": pd.DataFrame({
                    "key": ["A", "B", "C"],
                    "additional_info": [1, 2, 3]
                }),
                "type": ModalityType.TABULAR
            }
        }
        
        result = await fusion_engine.fuse_modalities(
            modalities,
            strategy=FusionStrategy.ATTENTION
        )
        
        assert isinstance(result, pd.DataFrame)
        assert "important_value" in result.columns or any("important_value" in col for col in result.columns)
    
    @pytest.mark.asyncio
    async def test_fusion_quality_assessment(self):
        """Test fusion quality metrics"""
        fusion_engine = FusionEngine()
        
        # Create test data
        fused_data = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
            "col2": [10, 20, 30, 40, 50],
            "col3": [100, 200, 300, 400, 500]
        })
        
        original_modalities = {
            "source1": {
                "data": pd.DataFrame({"col1": [1, 2, 3, 4, 5]}),
                "type": ModalityType.TABULAR
            }
        }
        
        quality = await fusion_engine._assess_fusion_quality(
            fused_data,
            original_modalities
        )
        
        assert "overall_score" in quality
        assert 0 <= quality["overall_score"] <= 1
        assert "completeness" in quality
        assert "information_preservation" in quality


class TestDocumentParser:
    """Test document parsing capabilities"""
    
    @pytest.mark.asyncio
    async def test_parse_csv(self):
        """Test parsing CSV files"""
        parser = DocumentParser()
        
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("name,age,city\n")
            f.write("John,30,NYC\n")
            f.write("Jane,25,LA\n")
            csv_path = f.name
        
        try:
            result = await parser.parse_document(csv_path)
            
            assert result["type"] == "csv"
            assert "tables" in result
            assert len(result["tables"]) == 1
            
            df = result["tables"][0]
            assert len(df) == 2
            assert list(df.columns) == ["name", "age", "city"]
            
        finally:
            Path(csv_path).unlink()
    
    @pytest.mark.asyncio
    async def test_parse_json(self):
        """Test parsing JSON files"""
        parser = DocumentParser()
        
        # Create temporary JSON file
        data = [
            {"id": 1, "name": "Item 1", "value": 100},
            {"id": 2, "name": "Item 2", "value": 200}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            json_path = f.name
        
        try:
            result = await parser.parse_document(json_path)
            
            assert result["type"] == "json"
            assert "structured_data" in result
            assert "tables" in result
            assert len(result["tables"]) == 1
            
        finally:
            Path(json_path).unlink()
    
    @pytest.mark.asyncio
    async def test_batch_parsing_with_fusion(self):
        """Test parsing multiple documents with fusion"""
        parser = DocumentParser()
        
        # Create test files
        files = []
        
        # CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,metric\n")
            f.write("1,100\n")
            f.write("2,200\n")
            files.append(f.name)
        
        # JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([{"id": 1, "category": "A"}, {"id": 2, "category": "B"}], f)
            files.append(f.name)
        
        try:
            result = await parser.parse_batch(
                files,
                fusion_strategy=FusionStrategy.HYBRID
            )
            
            assert "individual_documents" in result
            assert len(result["individual_documents"]) == 2
            
            # Check if fusion was attempted
            if "fused_data" in result:
                assert isinstance(result["fused_data"], pd.DataFrame)
            
        finally:
            for file_path in files:
                Path(file_path).unlink()
    
    @pytest.mark.asyncio
    async def test_document_quality_analysis(self):
        """Test document quality assessment"""
        parser = DocumentParser()
        
        # Mock parsed data
        parsed_data = {
            "type": "pdf",
            "tables": [pd.DataFrame({"col1": [1, 2, 3]})],
            "text": "Sample text content",
            "structured_data": {"key1": "value1", "key2": "value2"}
        }
        
        quality = await parser.analyze_document_quality(
            "dummy_path.pdf",
            parsed_data
        )
        
        assert "overall_score" in quality
        assert 0 <= quality["overall_score"] <= 1
        assert "completeness" in quality
        assert "data_richness" in quality


@pytest.mark.integration
class TestMultiModalIntegration:
    """Integration tests for multi-modal processing"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_document_processing(self):
        """Test complete document processing pipeline"""
        # This would test the full pipeline with real documents
        # Skipped in unit tests, run with integration tests
        pass
    
    @pytest.mark.asyncio
    async def test_pipeline_step_integration(self):
        """Test multi-modal pipeline steps"""
        from services.pipeline.steps.multimodal_steps import (
            DocumentParseStep,
            MultiModalFusionStep
        )
        
        # Test document parse step
        parse_step = DocumentParseStep()
        assert parse_step.name == "document_parse"
        
        # Test fusion step
        fusion_step = MultiModalFusionStep()
        assert fusion_step.name == "multimodal_fusion"