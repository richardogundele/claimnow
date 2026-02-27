# =============================================================================
# ClaimsNOW - Extractor Tests
# =============================================================================
# Unit tests for the field extraction module.
# Tests regex patterns and extraction accuracy.
# =============================================================================

import pytest
from src.extractor import (
    extract_fields,
    validate_extraction,
    _parse_date,
    _parse_currency,
    _normalize_vehicle_class
)


class TestDateParsing:
    """Tests for date parsing functionality."""
    
    def test_parse_uk_format_dd_mm_yyyy(self):
        """Test parsing DD/MM/YYYY format (UK standard)."""
        result = _parse_date("15/02/2026")
        assert result == "2026-02-15"
    
    def test_parse_hyphen_format(self):
        """Test parsing DD-MM-YYYY format."""
        result = _parse_date("15-02-2026")
        assert result == "2026-02-15"
    
    def test_parse_text_month(self):
        """Test parsing date with text month."""
        result = _parse_date("15 Feb 2026")
        assert result == "2026-02-15"
    
    def test_parse_full_month_name(self):
        """Test parsing date with full month name."""
        result = _parse_date("15 February 2026")
        assert result == "2026-02-15"
    
    def test_parse_invalid_returns_none(self):
        """Test that invalid dates return None."""
        result = _parse_date("not a date")
        assert result is None
    
    def test_parse_empty_returns_none(self):
        """Test that empty string returns None."""
        result = _parse_date("")
        assert result is None


class TestCurrencyParsing:
    """Tests for currency parsing functionality."""
    
    def test_parse_with_pound_sign(self):
        """Test parsing £89.00 format."""
        result = _parse_currency("£89.00")
        assert result == 89.00
    
    def test_parse_without_symbol(self):
        """Test parsing 89.00 without currency symbol."""
        result = _parse_currency("89.00")
        assert result == 89.00
    
    def test_parse_with_commas(self):
        """Test parsing £1,234.56 with thousand separators."""
        result = _parse_currency("£1,234.56")
        assert result == 1234.56
    
    def test_parse_gbp_prefix(self):
        """Test parsing GBP 89.00 format."""
        result = _parse_currency("GBP 89.00")
        assert result == 89.00
    
    def test_parse_invalid_returns_none(self):
        """Test that invalid currency returns None."""
        result = _parse_currency("not money")
        assert result is None


class TestVehicleClassNormalization:
    """Tests for vehicle class normalization."""
    
    def test_normalize_group_with_underscore(self):
        """Test normalizing GROUP_C format."""
        result = _normalize_vehicle_class("GROUP_C")
        assert result == "GROUP_C"
    
    def test_normalize_group_with_space(self):
        """Test normalizing 'Group C' format."""
        result = _normalize_vehicle_class("Group C")
        assert result == "GROUP_C"
    
    def test_normalize_single_letter(self):
        """Test normalizing single letter 'C'."""
        result = _normalize_vehicle_class("C")
        assert result == "GROUP_C"
    
    def test_normalize_lowercase(self):
        """Test normalizing lowercase 'group_c'."""
        result = _normalize_vehicle_class("group_c")
        assert result == "GROUP_C"
    
    def test_normalize_suv(self):
        """Test normalizing SUV."""
        result = _normalize_vehicle_class("SUV")
        assert result == "SUV"
    
    def test_normalize_mpv(self):
        """Test normalizing MPV."""
        result = _normalize_vehicle_class("mpv")
        assert result == "MPV"


class TestFieldExtraction:
    """Tests for the main field extraction function."""
    
    def test_extract_daily_rate_from_text(self):
        """Test extracting daily rate from document text."""
        text = """
        HIRE INVOICE
        Company: ABC Hire Ltd
        Daily Rate: £89.00
        Total: £1,246.00
        """
        
        extracted, confidence, missing = extract_fields(text)
        
        assert extracted.get("daily_rate") == 89.00
    
    def test_extract_total_claimed(self):
        """Test extracting total claimed amount."""
        text = """
        INVOICE TOTAL
        Hire charges: £1,000.00
        VAT: £200.00
        Total Due: £1,200.00
        """
        
        extracted, confidence, missing = extract_fields(text)
        
        assert extracted.get("total_claimed") == 1200.00
    
    def test_extract_vehicle_class(self):
        """Test extracting vehicle class."""
        text = """
        Vehicle Details:
        Group C - Ford Focus or similar
        Registration: AB12 CDE
        """
        
        extracted, confidence, missing = extract_fields(text)
        
        assert extracted.get("vehicle_class") == "GROUP_C"
    
    def test_extract_hire_dates(self):
        """Test extracting hire start and end dates."""
        text = """
        Hire Period:
        Start Date: 15/02/2026
        End Date: 01/03/2026
        Days: 15
        """
        
        extracted, confidence, missing = extract_fields(text)
        
        assert extracted.get("hire_start_date") == "2026-02-15"
        assert extracted.get("hire_end_date") == "2026-03-01"
    
    def test_extract_vehicle_registration(self):
        """Test extracting UK vehicle registration."""
        text = """
        Replacement Vehicle: Ford Focus
        Registration: AB12 CDE
        """
        
        extracted, confidence, missing = extract_fields(text)
        
        assert extracted.get("vehicle_registration") == "AB12CDE"
    
    def test_confidence_increases_with_more_fields(self):
        """Test that confidence increases with more extracted fields."""
        # Minimal text - low confidence
        minimal_text = "Invoice from company"
        _, low_confidence, _ = extract_fields(minimal_text)
        
        # Full text - high confidence
        full_text = """
        Daily Rate: £89.00
        Total: £1,246.00
        Vehicle Class: Group C
        Hire Period: 14 days
        Company: ABC Hire Ltd
        """
        _, high_confidence, _ = extract_fields(full_text)
        
        assert high_confidence > low_confidence


class TestExtractionValidation:
    """Tests for extraction validation."""
    
    def test_validate_reasonable_daily_rate(self):
        """Test validation passes for reasonable rates."""
        extracted = {
            "daily_rate": 89.00,
            "hire_days": 14,
            "total_claimed": 1246.00
        }
        
        is_valid, warnings = validate_extraction(extracted)
        
        # Should be valid with no major warnings
        assert is_valid or len(warnings) <= 1
    
    def test_validate_warns_on_high_rate(self):
        """Test validation warns on unusually high rates."""
        extracted = {
            "daily_rate": 600.00,  # Very high
            "hire_days": 14
        }
        
        is_valid, warnings = validate_extraction(extracted)
        
        # Should have warning about high rate
        assert any("high" in w.lower() for w in warnings)
    
    def test_validate_warns_on_total_mismatch(self):
        """Test validation warns when total doesn't match calculation."""
        extracted = {
            "daily_rate": 50.00,
            "hire_days": 10,
            "total_claimed": 1000.00  # Should be ~500
        }
        
        is_valid, warnings = validate_extraction(extracted)
        
        # Should have warning about total mismatch
        assert any("total" in w.lower() or "differ" in w.lower() for w in warnings)
