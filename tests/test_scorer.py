# =============================================================================
# ClaimsNOW - Scorer Tests
# =============================================================================
# Unit tests for the claim scoring and classification module.
# Tests threshold logic, confidence calculation, and verdict determination.
# =============================================================================

import pytest
from src.scorer import (
    score_claim,
    _determine_verdict,
    _calculate_overall_confidence,
    simulate_thresholds,
    summarize_batch_results
)
from src.config import Verdict


class TestVerdictDetermination:
    """Tests for verdict determination based on inflation ratio."""
    
    def test_fair_verdict_below_threshold(self):
        """Test FAIR verdict when ratio is below fair threshold."""
        # Ratio of 1.05 is below 1.1 threshold
        verdict = _determine_verdict(1.05)
        assert verdict == Verdict.FAIR
    
    def test_fair_verdict_at_boundary(self):
        """Test FAIR verdict at exact boundary (exclusive)."""
        # Ratio of 1.09 is still below 1.1
        verdict = _determine_verdict(1.09)
        assert verdict == Verdict.FAIR
    
    def test_inflated_verdict_above_fair(self):
        """Test POTENTIALLY_INFLATED when between thresholds."""
        # Ratio of 1.25 is between 1.1 and 1.4
        verdict = _determine_verdict(1.25)
        assert verdict == Verdict.POTENTIALLY_INFLATED
    
    def test_flagged_verdict_above_inflated(self):
        """Test FLAGGED when above inflated threshold."""
        # Ratio of 1.5 is above 1.4
        verdict = _determine_verdict(1.5)
        assert verdict == Verdict.FLAGGED
    
    def test_very_high_ratio_flagged(self):
        """Test FLAGGED for very high inflation ratios."""
        verdict = _determine_verdict(2.0)
        assert verdict == Verdict.FLAGGED
    
    def test_below_market_rate_fair(self):
        """Test FAIR when claimed rate is below market."""
        # Ratio of 0.8 means claimed is below market
        verdict = _determine_verdict(0.8)
        assert verdict == Verdict.FAIR


class TestClaimScoring:
    """Tests for the main scoring function."""
    
    def test_score_fair_claim(self):
        """Test scoring a fair claim."""
        extracted = {
            "daily_rate": 50.00,
            "total_claimed": 700.00,
            "hire_days": 14,
            "vehicle_class": "GROUP_C"
        }
        
        market = {
            "market_rate_low": 45.00,
            "market_rate_high": 65.00
        }
        
        result = score_claim(extracted, market, 0.95)
        
        assert result["verdict"] == Verdict.FAIR
        assert result["inflation_ratio"] < 1.1
    
    def test_score_inflated_claim(self):
        """Test scoring an inflated claim."""
        extracted = {
            "daily_rate": 80.00,  # Above market high of 65
            "total_claimed": 1120.00,
            "hire_days": 14,
            "vehicle_class": "GROUP_C"
        }
        
        market = {
            "market_rate_low": 45.00,
            "market_rate_high": 65.00
        }
        
        result = score_claim(extracted, market, 0.95)
        
        # 80/65 = 1.23, should be POTENTIALLY_INFLATED
        assert result["verdict"] == Verdict.POTENTIALLY_INFLATED
    
    def test_score_flagged_claim(self):
        """Test scoring a flagged claim."""
        extracted = {
            "daily_rate": 100.00,  # Way above market
            "total_claimed": 1400.00,
            "hire_days": 14,
            "vehicle_class": "GROUP_C"
        }
        
        market = {
            "market_rate_low": 45.00,
            "market_rate_high": 65.00
        }
        
        result = score_claim(extracted, market, 0.95)
        
        # 100/65 = 1.54, should be FLAGGED
        assert result["verdict"] == Verdict.FLAGGED
    
    def test_score_insufficient_data(self):
        """Test scoring when data is missing."""
        extracted = {
            "daily_rate": None,  # Missing rate
            "vehicle_class": "GROUP_C"
        }
        
        market = {
            "market_rate_low": 45.00,
            "market_rate_high": 65.00
        }
        
        result = score_claim(extracted, market, 0.3)
        
        assert result["verdict"] == Verdict.INSUFFICIENT_DATA
    
    def test_score_includes_excess_calculation(self):
        """Test that excess amounts are calculated correctly."""
        extracted = {
            "daily_rate": 80.00,
            "total_claimed": 1120.00,
            "hire_days": 14,
            "vehicle_class": "GROUP_C"
        }
        
        market = {
            "market_rate_low": 45.00,
            "market_rate_high": 65.00
        }
        
        result = score_claim(extracted, market, 0.95)
        
        # Daily excess should be 80 - 65 = 15
        assert result["excess_daily"] == 15.00
    
    def test_score_includes_recommendations(self):
        """Test that recommendations are included in result."""
        extracted = {
            "daily_rate": 80.00,
            "total_claimed": 1120.00,
            "hire_days": 14,
            "vehicle_class": "GROUP_C"
        }
        
        market = {
            "market_rate_low": 45.00,
            "market_rate_high": 65.00
        }
        
        result = score_claim(extracted, market, 0.95)
        
        assert "recommendations" in result
        assert len(result["recommendations"]) > 0


class TestConfidenceCalculation:
    """Tests for overall confidence calculation."""
    
    def test_high_extraction_confidence_maintained(self):
        """Test that high extraction confidence is reflected."""
        market = {"source": "database"}
        
        confidence = _calculate_overall_confidence(0.95, 1.05, market)
        
        # Should still be high
        assert confidence >= 0.85
    
    def test_fallback_source_reduces_confidence(self):
        """Test that fallback data source reduces confidence."""
        market_db = {"source": "database"}
        market_fallback = {"source": "fallback"}
        
        conf_db = _calculate_overall_confidence(0.95, 1.05, market_db)
        conf_fallback = _calculate_overall_confidence(0.95, 1.05, market_fallback)
        
        # Fallback should have lower confidence
        assert conf_fallback < conf_db
    
    def test_borderline_ratio_reduces_confidence(self):
        """Test that borderline ratios reduce confidence slightly."""
        market = {"source": "database"}
        
        # Clear case (ratio = 1.0)
        conf_clear = _calculate_overall_confidence(0.95, 1.0, market)
        
        # Borderline case (ratio = 1.1 - exactly at threshold)
        conf_borderline = _calculate_overall_confidence(0.95, 1.1, market)
        
        # Borderline should have slightly lower confidence
        assert conf_borderline <= conf_clear


class TestThresholdSimulation:
    """Tests for threshold simulation functionality."""
    
    def test_simulate_with_default_scenarios(self):
        """Test simulation with default threshold scenarios."""
        results = simulate_thresholds(1.25)
        
        # Should return multiple scenarios
        assert len(results) > 3
    
    def test_simulate_custom_scenarios(self):
        """Test simulation with custom threshold scenarios."""
        scenarios = [
            (1.0, 1.2),  # Very strict
            (1.2, 1.5),  # Very lenient
        ]
        
        results = simulate_thresholds(1.15, scenarios)
        
        assert len(results) == 2
        # 1.15 should be FLAGGED under strict, FAIR under lenient
        assert results[0]["verdict"] == Verdict.FLAGGED
        assert results[1]["verdict"] == Verdict.FAIR
    
    def test_simulate_preserves_ratio(self):
        """Test that simulation preserves the input ratio."""
        results = simulate_thresholds(1.33)
        
        for result in results:
            assert result["inflation_ratio"] == 1.33


class TestBatchSummary:
    """Tests for batch result summarization."""
    
    def test_summarize_counts_verdicts(self):
        """Test that summary correctly counts verdicts."""
        results = [
            {"verdict": Verdict.FAIR, "inflation_ratio": 1.0, "confidence_score": 0.9},
            {"verdict": Verdict.FAIR, "inflation_ratio": 1.05, "confidence_score": 0.85},
            {"verdict": Verdict.POTENTIALLY_INFLATED, "inflation_ratio": 1.2, "confidence_score": 0.9},
            {"verdict": Verdict.FLAGGED, "inflation_ratio": 1.5, "confidence_score": 0.95},
        ]
        
        summary = summarize_batch_results(results)
        
        assert summary["total_claims"] == 4
        assert summary["verdict_counts"][Verdict.FAIR] == 2
        assert summary["verdict_counts"][Verdict.POTENTIALLY_INFLATED] == 1
        assert summary["verdict_counts"][Verdict.FLAGGED] == 1
    
    def test_summarize_calculates_averages(self):
        """Test that summary calculates correct averages."""
        results = [
            {"verdict": Verdict.FAIR, "inflation_ratio": 1.0, "confidence_score": 0.8},
            {"verdict": Verdict.FAIR, "inflation_ratio": 1.2, "confidence_score": 1.0},
        ]
        
        summary = summarize_batch_results(results)
        
        # Average ratio should be 1.1
        assert summary["average_inflation_ratio"] == pytest.approx(1.1, rel=0.01)
        
        # Average confidence should be 0.9
        assert summary["average_confidence"] == pytest.approx(0.9, rel=0.01)
    
    def test_summarize_calculates_percentages(self):
        """Test that summary includes percentage calculations."""
        results = [
            {"verdict": Verdict.FAIR, "inflation_ratio": 1.0, "confidence_score": 0.9},
            {"verdict": Verdict.FLAGGED, "inflation_ratio": 1.5, "confidence_score": 0.9},
        ]
        
        summary = summarize_batch_results(results)
        
        # 50% fair, 50% flagged
        assert summary["fair_percentage"] == 50.0
        assert summary["flagged_percentage"] == 50.0
