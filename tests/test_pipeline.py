# =============================================================================
# ClaimsNOW - Pipeline Tests
# =============================================================================
# Integration tests for the full claim analysis pipeline.
# Tests end-to-end workflow and component coordination.
# =============================================================================

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.pipeline import (
    analyse_claim,
    validate_document,
    get_pipeline_status,
    _should_use_ai_extraction,
    _merge_extracted_data,
    _calculate_merged_confidence
)
from src.config import Verdict


class TestDocumentValidation:
    """Tests for document validation before analysis."""
    
    def test_validate_pdf_file(self):
        """Test validation of PDF file."""
        # Create mock PDF bytes (minimal valid size)
        pdf_bytes = b"%PDF-1.4" + b"0" * 2000
        
        is_valid, info, error = validate_document(pdf_bytes, "test.pdf")
        
        assert is_valid is True
        assert info["valid_format"] is True
        assert error is None
    
    def test_validate_png_file(self):
        """Test validation of PNG file."""
        png_bytes = b"\x89PNG" + b"0" * 2000
        
        is_valid, info, error = validate_document(png_bytes, "test.png")
        
        assert is_valid is True
    
    def test_reject_unsupported_format(self):
        """Test rejection of unsupported file formats."""
        doc_bytes = b"some content" * 100
        
        is_valid, info, error = validate_document(doc_bytes, "test.docx")
        
        assert is_valid is False
        assert "unsupported" in error.lower()
    
    def test_reject_oversized_file(self):
        """Test rejection of files exceeding size limit."""
        # Create 60MB file (exceeds 50MB limit)
        large_bytes = b"0" * (60 * 1024 * 1024)
        
        is_valid, info, error = validate_document(large_bytes, "large.pdf")
        
        assert is_valid is False
        assert "size" in error.lower() or "50MB" in error
    
    def test_warn_on_small_file(self):
        """Test warning for suspiciously small files."""
        small_bytes = b"0" * 500  # Less than 1KB
        
        is_valid, info, error = validate_document(small_bytes, "tiny.pdf")
        
        # Should be valid but with warning
        assert len(info.get("warnings", [])) > 0


class TestAIExtractionDecision:
    """Tests for deciding whether to use AI extraction."""
    
    def test_use_ai_when_confidence_low(self):
        """Test AI is used when extraction confidence is low."""
        should_use = _should_use_ai_extraction(
            confidence=0.5,  # Below 0.7 threshold
            missing_fields=[],
            options={}
        )
        
        assert should_use is True
    
    def test_skip_ai_when_confidence_high(self):
        """Test AI is skipped when confidence is sufficient."""
        should_use = _should_use_ai_extraction(
            confidence=0.9,  # Above threshold
            missing_fields=[],
            options={}
        )
        
        assert should_use is False
    
    def test_use_ai_when_required_fields_missing(self):
        """Test AI is used when required fields are missing."""
        should_use = _should_use_ai_extraction(
            confidence=0.9,  # High confidence
            missing_fields=["daily_rate"],  # But missing critical field
            options={}
        )
        
        assert should_use is True
    
    def test_skip_ai_when_disabled(self):
        """Test AI is skipped when explicitly disabled."""
        should_use = _should_use_ai_extraction(
            confidence=0.5,  # Low confidence
            missing_fields=["daily_rate"],  # Missing fields
            options={"skip_ai": True}  # But AI disabled
        )
        
        assert should_use is False
    
    def test_force_ai_overrides_all(self):
        """Test force_ai option always uses AI."""
        should_use = _should_use_ai_extraction(
            confidence=1.0,  # Perfect confidence
            missing_fields=[],  # Nothing missing
            options={"force_ai": True}  # Force AI anyway
        )
        
        assert should_use is True


class TestDataMerging:
    """Tests for merging regex and AI extracted data."""
    
    def test_merge_fills_gaps(self):
        """Test that AI data fills gaps in regex data."""
        regex_data = {
            "daily_rate": 89.00,
            "vehicle_class": None,  # Missing
            "hire_days": 14
        }
        
        ai_data = {
            "daily_rate": 88.00,  # Slightly different
            "vehicle_class": "GROUP_C",  # Has this
            "hire_days": 14
        }
        
        merged = _merge_extracted_data(regex_data, ai_data)
        
        # Regex value should be kept for daily_rate
        assert merged["daily_rate"] == 89.00
        
        # AI value should fill the gap for vehicle_class
        assert merged["vehicle_class"] == "GROUP_C"
    
    def test_merge_prefers_regex(self):
        """Test that regex values are preferred over AI."""
        regex_data = {
            "daily_rate": 89.00,
            "vehicle_class": "GROUP_D"
        }
        
        ai_data = {
            "daily_rate": 85.00,  # Different
            "vehicle_class": "GROUP_C"  # Different
        }
        
        merged = _merge_extracted_data(regex_data, ai_data)
        
        # Regex values should be kept
        assert merged["daily_rate"] == 89.00
        assert merged["vehicle_class"] == "GROUP_D"
    
    def test_merge_improves_unknown_vehicle_class(self):
        """Test that AI can improve UNKNOWN vehicle class."""
        regex_data = {
            "vehicle_class": "UNKNOWN"  # Regex couldn't determine
        }
        
        ai_data = {
            "vehicle_class": "GROUP_C"  # AI figured it out
        }
        
        merged = _merge_extracted_data(regex_data, ai_data)
        
        # AI should improve the UNKNOWN value
        assert merged["vehicle_class"] == "GROUP_C"


class TestMergedConfidence:
    """Tests for confidence calculation on merged data."""
    
    def test_confidence_increases_with_fields(self):
        """Test that confidence increases with more fields."""
        minimal_data = {
            "daily_rate": None,
            "vehicle_class": None,
            "hire_days": None
        }
        
        full_data = {
            "daily_rate": 89.00,
            "vehicle_class": "GROUP_C",
            "hire_days": 14,
            "total_claimed": 1246.00,
            "hire_company": "ABC Hire Ltd",
            "hire_start_date": "2026-02-15",
            "hire_end_date": "2026-03-01"
        }
        
        minimal_conf = _calculate_merged_confidence(minimal_data)
        full_conf = _calculate_merged_confidence(full_data)
        
        assert full_conf > minimal_conf
    
    def test_confidence_capped_at_one(self):
        """Test that confidence is capped at 1.0."""
        complete_data = {
            "daily_rate": 89.00,
            "vehicle_class": "GROUP_C",
            "hire_days": 14,
            "total_claimed": 1246.00,
            "hire_company": "ABC Hire Ltd",
            "hire_start_date": "2026-02-15",
            "hire_end_date": "2026-03-01"
        }
        
        confidence = _calculate_merged_confidence(complete_data)
        
        assert confidence <= 1.0


class TestPipelineStatus:
    """Tests for pipeline status reporting."""
    
    def test_status_includes_version(self):
        """Test that status includes pipeline version."""
        status = get_pipeline_status()
        
        assert "pipeline_version" in status
        assert status["pipeline_version"] is not None
    
    def test_status_includes_components(self):
        """Test that status includes component statuses."""
        status = get_pipeline_status()
        
        assert "components" in status
        # Should check S3, Textract, DynamoDB, Bedrock
        assert "s3" in status["components"]
        assert "textract" in status["components"]
        assert "dynamodb" in status["components"]
        assert "bedrock" in status["components"]
    
    def test_status_includes_configuration(self):
        """Test that status includes configuration info."""
        status = get_pipeline_status()
        
        assert "configuration" in status
        assert "threshold_fair" in status["configuration"]
        assert "threshold_inflated" in status["configuration"]


class TestFullPipeline:
    """Integration tests for the full pipeline."""
    
    @pytest.mark.skip(reason="Requires AWS credentials")
    def test_full_analysis_flow(self):
        """Test complete analysis from upload to verdict."""
        # This would require AWS setup
        pass
    
    def test_pipeline_handles_errors_gracefully(self):
        """Test that pipeline returns proper error on failure."""
        # Empty bytes should fail
        success, result, error = analyse_claim(b"", "empty.pdf")
        
        # Should fail but not crash
        assert success is False
        assert error is not None
