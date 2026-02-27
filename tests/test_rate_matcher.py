# =============================================================================
# ClaimsNOW - Rate Matcher Tests
# =============================================================================
# Unit tests for the rate matching engine.
# Tests market rate lookup, comparison logic, and fallback handling.
# =============================================================================

import pytest
from src.rate_matcher import (
    find_market_rate,
    compare_rate,
    adjust_rate_for_period,
    validate_rate_data,
    generate_rate_summary,
    _normalize_vehicle_class,
    _normalize_region
)
from src.config import VehicleClass, Region


class TestVehicleClassNormalization:
    """Tests for vehicle class normalization in rate matcher."""
    
    def test_normalize_valid_class(self):
        """Test normalizing already valid class."""
        result = _normalize_vehicle_class("GROUP_C")
        assert result == "GROUP_C"
    
    def test_normalize_with_space(self):
        """Test normalizing class with space."""
        result = _normalize_vehicle_class("GROUP C")
        assert result == "GROUP_C"
    
    def test_normalize_lowercase(self):
        """Test normalizing lowercase input."""
        result = _normalize_vehicle_class("group_c")
        assert result == "GROUP_C"
    
    def test_normalize_single_letter(self):
        """Test normalizing single letter."""
        result = _normalize_vehicle_class("C")
        assert result == "GROUP_C"
    
    def test_normalize_suv(self):
        """Test normalizing SUV."""
        result = _normalize_vehicle_class("suv")
        assert result == VehicleClass.SUV
    
    def test_normalize_unknown(self):
        """Test normalizing unknown class."""
        result = _normalize_vehicle_class("SPACESHIP")
        assert result == VehicleClass.UNKNOWN
    
    def test_normalize_empty(self):
        """Test normalizing empty string."""
        result = _normalize_vehicle_class("")
        assert result == VehicleClass.UNKNOWN
    
    def test_normalize_none(self):
        """Test normalizing None."""
        result = _normalize_vehicle_class(None)
        assert result == VehicleClass.UNKNOWN


class TestRegionNormalization:
    """Tests for region normalization."""
    
    def test_normalize_valid_region(self):
        """Test normalizing already valid region."""
        result = _normalize_region("LONDON")
        assert result == Region.LONDON
    
    def test_normalize_lowercase(self):
        """Test normalizing lowercase region."""
        result = _normalize_region("london")
        assert result == "LONDON"
    
    def test_normalize_with_space(self):
        """Test normalizing region with space."""
        result = _normalize_region("NORTH WEST")
        assert result == "NORTH_WEST"
    
    def test_normalize_unknown(self):
        """Test normalizing unknown region."""
        result = _normalize_region("MARS")
        assert result == Region.UNKNOWN
    
    def test_normalize_empty(self):
        """Test normalizing empty string."""
        result = _normalize_region("")
        assert result == Region.UNKNOWN


class TestRateComparison:
    """Tests for rate comparison logic."""
    
    def test_rate_within_range(self):
        """Test comparison when rate is within market range."""
        result = compare_rate(55.00, 45.00, 65.00)
        
        assert result["within_range"] is True
        assert result["above_range"] is False
        assert result["below_range"] is False
        assert result["excess_amount"] == 0
    
    def test_rate_below_range(self):
        """Test comparison when rate is below market minimum."""
        result = compare_rate(40.00, 45.00, 65.00)
        
        assert result["below_range"] is True
        assert result["within_range"] is False
        assert result["above_range"] is False
    
    def test_rate_above_range(self):
        """Test comparison when rate is above market maximum."""
        result = compare_rate(80.00, 45.00, 65.00)
        
        assert result["above_range"] is True
        assert result["within_range"] is False
        assert result["below_range"] is False
    
    def test_inflation_ratio_calculation(self):
        """Test inflation ratio calculation."""
        result = compare_rate(80.00, 45.00, 65.00)
        
        # 80 / 65 = 1.23
        expected_ratio = 80.00 / 65.00
        assert result["inflation_ratio"] == pytest.approx(expected_ratio, rel=0.01)
    
    def test_excess_amount_calculation(self):
        """Test excess amount calculation."""
        result = compare_rate(80.00, 45.00, 65.00)
        
        # Excess is 80 - 65 = 15
        assert result["excess_amount"] == 15.00
    
    def test_excess_percentage_calculation(self):
        """Test excess percentage calculation."""
        result = compare_rate(80.00, 45.00, 65.00)
        
        # Excess percentage: (15/65) * 100 = 23.08%
        assert result["excess_percentage"] == pytest.approx(23.08, rel=0.1)
    
    def test_market_midpoint_calculation(self):
        """Test market midpoint calculation."""
        result = compare_rate(55.00, 45.00, 65.00)
        
        # Midpoint: (45 + 65) / 2 = 55
        assert result["market_midpoint"] == 55.00


class TestPeriodAdjustment:
    """Tests for hire period rate adjustments."""
    
    def test_no_discount_short_period(self):
        """Test no discount for short hire periods (1-7 days)."""
        result = adjust_rate_for_period(50.00, 70.00, 5)
        
        # No discount for short periods
        assert result["high"] == 70.00
        assert result["discount_applied"] == 0.0
    
    def test_5_percent_discount_medium_period(self):
        """Test 5% discount for medium periods (8-14 days)."""
        result = adjust_rate_for_period(50.00, 70.00, 10)
        
        # 5% off high: 70 * 0.95 = 66.50
        assert result["high"] == pytest.approx(66.50, rel=0.01)
        assert result["discount_applied"] == 0.05
    
    def test_10_percent_discount_longer_period(self):
        """Test 10% discount for longer periods (15-30 days)."""
        result = adjust_rate_for_period(50.00, 70.00, 21)
        
        # 10% off high: 70 * 0.90 = 63.00
        assert result["high"] == pytest.approx(63.00, rel=0.01)
        assert result["discount_applied"] == 0.10
    
    def test_15_percent_discount_long_period(self):
        """Test 15% discount for very long periods (31+ days)."""
        result = adjust_rate_for_period(50.00, 70.00, 45)
        
        # 15% off high: 70 * 0.85 = 59.50
        assert result["high"] == pytest.approx(59.50, rel=0.01)
        assert result["discount_applied"] == 0.15
    
    def test_low_rate_unchanged(self):
        """Test that low rate is never adjusted."""
        result = adjust_rate_for_period(50.00, 70.00, 45)
        
        # Low should always stay the same
        assert result["low"] == 50.00


class TestRateDataValidation:
    """Tests for rate data validation."""
    
    def test_valid_rate_data(self):
        """Test validation of correct rate data."""
        rate_data = {
            "market_rate_low": 45.00,
            "market_rate_high": 65.00,
            "source_year": 2026
        }
        
        is_valid, issues = validate_rate_data(rate_data)
        
        assert is_valid is True
        assert len(issues) == 0
    
    def test_invalid_negative_rates(self):
        """Test validation catches negative rates."""
        rate_data = {
            "market_rate_low": -10.00,
            "market_rate_high": 65.00
        }
        
        is_valid, issues = validate_rate_data(rate_data)
        
        assert is_valid is False
        assert any("positive" in issue.lower() for issue in issues)
    
    def test_invalid_low_exceeds_high(self):
        """Test validation catches low > high."""
        rate_data = {
            "market_rate_low": 100.00,
            "market_rate_high": 50.00
        }
        
        is_valid, issues = validate_rate_data(rate_data)
        
        assert is_valid is False
        assert any("exceeds" in issue.lower() for issue in issues)
    
    def test_warns_on_old_data(self):
        """Test validation warns on outdated data."""
        rate_data = {
            "market_rate_low": 45.00,
            "market_rate_high": 65.00,
            "source_year": 2018  # Old data
        }
        
        is_valid, issues = validate_rate_data(rate_data)
        
        assert any("outdated" in issue.lower() for issue in issues)


class TestRateSummary:
    """Tests for rate summary generation."""
    
    def test_summary_for_within_range(self):
        """Test summary text for rate within market range."""
        rate_data = {
            "vehicle_class": "GROUP_C",
            "region": "LONDON",
            "market_rate_low": 45.00,
            "market_rate_high": 65.00,
            "source": "database",
            "source_year": 2026
        }
        
        comparison = {
            "within_range": True,
            "above_range": False,
            "below_range": False,
            "inflation_ratio": 0.85,
            "excess_percentage": 0
        }
        
        summary = generate_rate_summary(55.00, rate_data, comparison)
        
        assert "within" in summary.lower()
        assert "fair" in summary.lower()
    
    def test_summary_for_above_range(self):
        """Test summary text for rate above market range."""
        rate_data = {
            "vehicle_class": "GROUP_C",
            "region": "LONDON",
            "market_rate_low": 45.00,
            "market_rate_high": 65.00,
            "source": "database",
            "source_year": 2026
        }
        
        comparison = {
            "within_range": False,
            "above_range": True,
            "below_range": False,
            "inflation_ratio": 1.23,
            "excess_percentage": 23.0
        }
        
        summary = generate_rate_summary(80.00, rate_data, comparison)
        
        assert "exceeds" in summary.lower()
        assert "23" in summary  # Excess percentage
