# =============================================================================
# ClaimsNOW - Claim Scoring and Classification Module
# =============================================================================
# This module determines whether an insurance claim is fair or inflated.
#
# The scorer takes:
# 1. Extracted claim data (daily rate, total, etc.)
# 2. Market rate comparison data
# 3. Extraction confidence score
#
# And produces:
# - A verdict (FAIR, POTENTIALLY_INFLATED, or FLAGGED)
# - A confidence score
# - Detailed scoring breakdown
#
# Thresholds are configurable in config.py to allow business rule tuning.
# =============================================================================

import logging
from typing import Dict, Optional, Tuple, Any, List
from datetime import datetime

from src.config import settings, Verdict

# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# Main Scoring Function
# =============================================================================

def score_claim(
    extracted_data: Dict[str, Any],
    market_rate_data: Dict[str, Any],
    extraction_confidence: float
) -> Dict[str, Any]:
    """
    Score a claim and determine the verdict.
    
    This is the main scoring function. It:
    1. Calculates the inflation ratio
    2. Applies threshold rules to determine verdict
    3. Adjusts confidence based on data quality
    4. Returns a complete scoring result
    
    Args:
        extracted_data: Fields extracted from the document:
            - daily_rate: The claimed daily rate
            - total_claimed: Total amount claimed
            - hire_days: Number of hire days
            - vehicle_class: Vehicle category
        market_rate_data: Market rate comparison:
            - market_rate_low: Lower bound of fair rate
            - market_rate_high: Upper bound of fair rate
        extraction_confidence: Confidence in the extraction (0.0-1.0)
    
    Returns:
        dict: Complete scoring result:
            - verdict: FAIR, POTENTIALLY_INFLATED, FLAGGED, or INSUFFICIENT_DATA
            - inflation_ratio: Claimed rate vs market high
            - confidence_score: Overall confidence in the result
            - daily_rate_claimed: The claimed daily rate
            - daily_rate_market_high: Market upper bound
            - total_claimed: Total amount claimed
            - total_market_estimate: Expected total at market rate
            - excess_daily: Daily amount above market
            - excess_total: Total amount above market
            - scoring_factors: Detailed breakdown of scoring
            - recommendations: Suggested actions
    
    Example:
        >>> result = score_claim(extracted, market_rate, 0.95)
        >>> print(f"Verdict: {result['verdict']}")
        >>> print(f"Inflation ratio: {result['inflation_ratio']:.2f}")
    """
    # Extract key values
    daily_rate = extracted_data.get("daily_rate")
    total_claimed = extracted_data.get("total_claimed")
    hire_days = extracted_data.get("hire_days")
    
    market_rate_high = market_rate_data.get("market_rate_high")
    market_rate_low = market_rate_data.get("market_rate_low")
    
    # Check if we have enough data to score
    if not daily_rate or not market_rate_high:
        return _create_insufficient_data_result(extracted_data, extraction_confidence)
    
    # Calculate inflation ratio
    # This is the key metric: claimed rate / market upper bound
    inflation_ratio = daily_rate / market_rate_high
    
    # Calculate excess amounts
    excess_daily = max(0, daily_rate - market_rate_high)
    
    # Estimate market-rate total
    if hire_days:
        total_market_estimate = market_rate_high * hire_days
        excess_total = max(0, (total_claimed or 0) - total_market_estimate)
    else:
        total_market_estimate = None
        excess_total = None
    
    # Determine verdict based on inflation ratio
    verdict = _determine_verdict(inflation_ratio)
    
    # Calculate overall confidence
    confidence_score = _calculate_overall_confidence(
        extraction_confidence,
        inflation_ratio,
        market_rate_data
    )
    
    # Generate scoring factors breakdown
    scoring_factors = _generate_scoring_factors(
        daily_rate,
        market_rate_low,
        market_rate_high,
        inflation_ratio,
        hire_days,
        extraction_confidence
    )
    
    # Generate recommendations
    recommendations = _generate_recommendations(verdict, inflation_ratio, confidence_score)
    
    result = {
        # Core verdict
        "verdict": verdict,
        "inflation_ratio": round(inflation_ratio, 3),
        "confidence_score": round(confidence_score, 3),
        
        # Rate details
        "daily_rate_claimed": daily_rate,
        "daily_rate_market_low": market_rate_low,
        "daily_rate_market_high": market_rate_high,
        
        # Total details
        "total_claimed": total_claimed,
        "total_market_estimate": round(total_market_estimate, 2) if total_market_estimate else None,
        
        # Excess calculations
        "excess_daily": round(excess_daily, 2),
        "excess_total": round(excess_total, 2) if excess_total else None,
        "excess_percentage": round((inflation_ratio - 1) * 100, 1) if inflation_ratio > 1 else 0,
        
        # Detailed breakdown
        "scoring_factors": scoring_factors,
        "recommendations": recommendations,
        
        # Metadata
        "scored_at": datetime.utcnow().isoformat(),
        "thresholds_used": {
            "fair": settings.THRESHOLD_FAIR,
            "inflated": settings.THRESHOLD_INFLATED
        }
    }
    
    logger.info(f"Claim scored: {verdict} (ratio={inflation_ratio:.2f}, confidence={confidence_score:.2f})")
    
    return result


def _determine_verdict(inflation_ratio: float) -> str:
    """
    Determine verdict based on inflation ratio and configured thresholds.
    
    Thresholds (from config):
    - Below THRESHOLD_FAIR (1.1): FAIR
    - Between THRESHOLD_FAIR and THRESHOLD_INFLATED: POTENTIALLY_INFLATED
    - Above THRESHOLD_INFLATED (1.4): FLAGGED
    
    Args:
        inflation_ratio: Claimed rate / market rate high.
    
    Returns:
        str: Verdict constant (FAIR, POTENTIALLY_INFLATED, or FLAGGED)
    """
    if inflation_ratio < settings.THRESHOLD_FAIR:
        return Verdict.FAIR
    elif inflation_ratio < settings.THRESHOLD_INFLATED:
        return Verdict.POTENTIALLY_INFLATED
    else:
        return Verdict.FLAGGED


def _calculate_overall_confidence(
    extraction_confidence: float,
    inflation_ratio: float,
    market_rate_data: Dict[str, Any]
) -> float:
    """
    Calculate overall confidence in the scoring result.
    
    Confidence is reduced when:
    - Extraction confidence is low
    - Market rate data is from fallback source
    - Inflation ratio is very close to threshold boundaries
    
    Args:
        extraction_confidence: Confidence in extracted data.
        inflation_ratio: The calculated inflation ratio.
        market_rate_data: Market rate information.
    
    Returns:
        float: Overall confidence score (0.0-1.0)
    """
    # Start with extraction confidence
    confidence = extraction_confidence
    
    # Reduce confidence if using fallback rates
    if market_rate_data.get("source") == "fallback":
        confidence *= 0.8  # 20% reduction for fallback data
    
    # Reduce confidence if close to threshold boundaries
    # This reflects uncertainty in borderline cases
    threshold_proximity = _calculate_threshold_proximity(inflation_ratio)
    if threshold_proximity < 0.05:  # Within 5% of a threshold
        confidence *= 0.9  # 10% reduction for borderline cases
    
    # Ensure confidence is within bounds
    return max(0.0, min(1.0, confidence))


def _calculate_threshold_proximity(inflation_ratio: float) -> float:
    """
    Calculate how close the inflation ratio is to threshold boundaries.
    
    Returns the distance to the nearest threshold as a proportion.
    
    Args:
        inflation_ratio: The inflation ratio to check.
    
    Returns:
        float: Distance to nearest threshold (larger = further from boundary)
    """
    thresholds = [1.0, settings.THRESHOLD_FAIR, settings.THRESHOLD_INFLATED]
    
    min_distance = float('inf')
    for threshold in thresholds:
        distance = abs(inflation_ratio - threshold)
        min_distance = min(min_distance, distance)
    
    return min_distance


def _create_insufficient_data_result(
    extracted_data: Dict[str, Any],
    extraction_confidence: float
) -> Dict[str, Any]:
    """
    Create a result for when there's not enough data to score.
    
    Args:
        extracted_data: Whatever was extracted.
        extraction_confidence: Extraction confidence score.
    
    Returns:
        dict: Scoring result indicating insufficient data.
    """
    return {
        "verdict": Verdict.INSUFFICIENT_DATA,
        "inflation_ratio": None,
        "confidence_score": extraction_confidence,
        "daily_rate_claimed": extracted_data.get("daily_rate"),
        "daily_rate_market_low": None,
        "daily_rate_market_high": None,
        "total_claimed": extracted_data.get("total_claimed"),
        "total_market_estimate": None,
        "excess_daily": None,
        "excess_total": None,
        "excess_percentage": None,
        "scoring_factors": [
            {
                "factor": "insufficient_data",
                "description": "Could not determine market rate or extract daily rate",
                "impact": "critical",
                "recommendation": "Manual review required"
            }
        ],
        "recommendations": [
            "Manual review required - insufficient data for automated scoring",
            "Check document quality and try re-uploading",
            "Verify the document contains hire rate information"
        ],
        "scored_at": datetime.utcnow().isoformat(),
        "thresholds_used": {
            "fair": settings.THRESHOLD_FAIR,
            "inflated": settings.THRESHOLD_INFLATED
        }
    }


# =============================================================================
# Scoring Factors Breakdown
# =============================================================================

def _generate_scoring_factors(
    daily_rate: float,
    market_low: float,
    market_high: float,
    inflation_ratio: float,
    hire_days: Optional[int],
    extraction_confidence: float
) -> List[Dict[str, Any]]:
    """
    Generate detailed breakdown of factors that influenced the score.
    
    This provides transparency into the scoring decision, which is
    critical for governance and reviewer trust.
    
    Args:
        daily_rate: Claimed daily rate.
        market_low: Market rate lower bound.
        market_high: Market rate upper bound.
        inflation_ratio: Calculated inflation ratio.
        hire_days: Number of hire days.
        extraction_confidence: Confidence in extraction.
    
    Returns:
        list: Scoring factors with descriptions and impacts.
    """
    factors = []
    
    # Factor 1: Rate position relative to market
    if daily_rate <= market_low:
        factors.append({
            "factor": "rate_position",
            "description": f"Daily rate (£{daily_rate:.2f}) is at or below market minimum (£{market_low:.2f})",
            "impact": "positive",
            "weight": 0.4
        })
    elif daily_rate <= market_high:
        factors.append({
            "factor": "rate_position",
            "description": f"Daily rate (£{daily_rate:.2f}) is within market range (£{market_low:.2f}-£{market_high:.2f})",
            "impact": "positive",
            "weight": 0.3
        })
    else:
        excess_pct = (inflation_ratio - 1) * 100
        factors.append({
            "factor": "rate_position",
            "description": f"Daily rate (£{daily_rate:.2f}) exceeds market maximum (£{market_high:.2f}) by {excess_pct:.1f}%",
            "impact": "negative",
            "weight": 0.4
        })
    
    # Factor 2: Inflation ratio severity
    if inflation_ratio < 1.0:
        severity = "below_market"
        impact = "positive"
        desc = "Rate is below market average"
    elif inflation_ratio < settings.THRESHOLD_FAIR:
        severity = "acceptable"
        impact = "positive"
        desc = "Rate is within acceptable range"
    elif inflation_ratio < settings.THRESHOLD_INFLATED:
        severity = "elevated"
        impact = "neutral"
        desc = "Rate is elevated but not extreme"
    else:
        severity = "excessive"
        impact = "negative"
        desc = "Rate significantly exceeds market norms"
    
    factors.append({
        "factor": "inflation_severity",
        "description": desc,
        "severity": severity,
        "inflation_ratio": round(inflation_ratio, 3),
        "impact": impact,
        "weight": 0.3
    })
    
    # Factor 3: Hire duration impact
    if hire_days:
        if hire_days <= 7:
            duration_factor = "short_term"
            desc = f"Short hire period ({hire_days} days) - standard rates expected"
        elif hire_days <= 30:
            duration_factor = "medium_term"
            desc = f"Medium hire period ({hire_days} days) - some discount expected"
        else:
            duration_factor = "long_term"
            desc = f"Long hire period ({hire_days} days) - significant discount expected"
        
        factors.append({
            "factor": "hire_duration",
            "description": desc,
            "duration_factor": duration_factor,
            "hire_days": hire_days,
            "impact": "informational",
            "weight": 0.15
        })
    
    # Factor 4: Data quality
    if extraction_confidence >= 0.9:
        quality = "high"
        desc = "High confidence in extracted data"
        impact = "positive"
    elif extraction_confidence >= 0.7:
        quality = "medium"
        desc = "Medium confidence in extracted data"
        impact = "neutral"
    else:
        quality = "low"
        desc = "Low confidence in extracted data - review recommended"
        impact = "negative"
    
    factors.append({
        "factor": "data_quality",
        "description": desc,
        "quality": quality,
        "confidence": round(extraction_confidence, 3),
        "impact": impact,
        "weight": 0.15
    })
    
    return factors


# =============================================================================
# Recommendations Generation
# =============================================================================

def _generate_recommendations(
    verdict: str,
    inflation_ratio: float,
    confidence_score: float
) -> List[str]:
    """
    Generate action recommendations based on scoring result.
    
    Provides clear guidance for the reviewer on what to do next.
    
    Args:
        verdict: The scoring verdict.
        inflation_ratio: The inflation ratio.
        confidence_score: Overall confidence.
    
    Returns:
        list: Recommended actions for the reviewer.
    """
    recommendations = []
    
    # Recommendations based on verdict
    if verdict == Verdict.FAIR:
        recommendations.append("Claim appears fair based on market comparison")
        recommendations.append("Standard processing can proceed")
        
    elif verdict == Verdict.POTENTIALLY_INFLATED:
        recommendations.append("Claim warrants closer review")
        recommendations.append("Consider requesting breakdown of charges")
        recommendations.append("Check for special circumstances justifying higher rate")
        
    elif verdict == Verdict.FLAGGED:
        recommendations.append("PRIORITY REVIEW REQUIRED")
        recommendations.append(f"Rate exceeds market by {(inflation_ratio - 1) * 100:.0f}%")
        recommendations.append("Request detailed justification from hire company")
        recommendations.append("Consider challenge or negotiation")
        
    elif verdict == Verdict.INSUFFICIENT_DATA:
        recommendations.append("Manual review required due to incomplete data")
        recommendations.append("Request additional documentation")
        recommendations.append("Verify document upload quality")
    
    # Additional recommendations based on confidence
    if confidence_score < 0.7:
        recommendations.append("LOW CONFIDENCE: Verify extracted values manually")
    
    if confidence_score < 0.5:
        recommendations.append("Consider re-processing with clearer document")
    
    return recommendations


# =============================================================================
# Batch Scoring Functions
# =============================================================================

def score_batch(
    claims: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Score multiple claims in a batch.
    
    Useful for processing multiple documents at once.
    
    Args:
        claims: List of claims, each containing:
            - extracted_data: Extracted claim fields
            - market_rate_data: Market rate info
            - extraction_confidence: Extraction confidence
    
    Returns:
        list: List of scoring results, one per claim.
    """
    results = []
    
    for i, claim in enumerate(claims):
        try:
            result = score_claim(
                claim.get("extracted_data", {}),
                claim.get("market_rate_data", {}),
                claim.get("extraction_confidence", 0.5)
            )
            result["batch_index"] = i
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error scoring claim {i}: {str(e)}")
            results.append({
                "batch_index": i,
                "verdict": Verdict.INSUFFICIENT_DATA,
                "error": str(e)
            })
    
    return results


def summarize_batch_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary statistics for batch scoring results.
    
    Args:
        results: List of scoring results from score_batch.
    
    Returns:
        dict: Summary statistics:
            - total_claims: Number of claims processed
            - verdict_counts: Count by verdict type
            - average_inflation_ratio: Mean inflation ratio
            - high_risk_count: Claims flagged for review
            - confidence_distribution: Breakdown by confidence level
    """
    total = len(results)
    
    # Count verdicts
    verdict_counts = {
        Verdict.FAIR: 0,
        Verdict.POTENTIALLY_INFLATED: 0,
        Verdict.FLAGGED: 0,
        Verdict.INSUFFICIENT_DATA: 0
    }
    
    inflation_ratios = []
    confidence_scores = []
    
    for result in results:
        verdict = result.get("verdict", Verdict.INSUFFICIENT_DATA)
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        
        if result.get("inflation_ratio") is not None:
            inflation_ratios.append(result["inflation_ratio"])
        
        if result.get("confidence_score") is not None:
            confidence_scores.append(result["confidence_score"])
    
    # Calculate averages
    avg_inflation = sum(inflation_ratios) / len(inflation_ratios) if inflation_ratios else None
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else None
    
    # Count high-risk claims (FLAGGED + low confidence POTENTIALLY_INFLATED)
    high_risk = verdict_counts[Verdict.FLAGGED]
    
    return {
        "total_claims": total,
        "verdict_counts": verdict_counts,
        "average_inflation_ratio": round(avg_inflation, 3) if avg_inflation else None,
        "average_confidence": round(avg_confidence, 3) if avg_confidence else None,
        "high_risk_count": high_risk,
        "fair_percentage": round(verdict_counts[Verdict.FAIR] / total * 100, 1) if total > 0 else 0,
        "flagged_percentage": round(verdict_counts[Verdict.FLAGGED] / total * 100, 1) if total > 0 else 0
    }


# =============================================================================
# Threshold Simulation
# =============================================================================

def simulate_thresholds(
    inflation_ratio: float,
    threshold_scenarios: List[Tuple[float, float]] = None
) -> List[Dict[str, Any]]:
    """
    Simulate how different thresholds would affect the verdict.
    
    Useful for understanding sensitivity to threshold settings
    and for business rule tuning.
    
    Args:
        inflation_ratio: The inflation ratio to test.
        threshold_scenarios: List of (fair_threshold, inflated_threshold) tuples.
            Defaults to a range of common configurations.
    
    Returns:
        list: Results for each threshold scenario.
    
    Example:
        >>> scenarios = simulate_thresholds(1.25)
        >>> for s in scenarios:
        ...     print(f"Fair={s['fair_threshold']}: {s['verdict']}")
    """
    if threshold_scenarios is None:
        # Default scenarios: various threshold combinations
        threshold_scenarios = [
            (1.05, 1.20),  # Strict
            (1.10, 1.30),  # Moderate-strict
            (1.10, 1.40),  # Current default
            (1.15, 1.40),  # Moderate-lenient
            (1.20, 1.50),  # Lenient
        ]
    
    results = []
    
    for fair_threshold, inflated_threshold in threshold_scenarios:
        if inflation_ratio < fair_threshold:
            verdict = Verdict.FAIR
        elif inflation_ratio < inflated_threshold:
            verdict = Verdict.POTENTIALLY_INFLATED
        else:
            verdict = Verdict.FLAGGED
        
        results.append({
            "fair_threshold": fair_threshold,
            "inflated_threshold": inflated_threshold,
            "verdict": verdict,
            "inflation_ratio": inflation_ratio
        })
    
    return results
