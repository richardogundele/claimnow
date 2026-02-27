# =============================================================================
# ClaimsNOW - Explainability Layer Module
# =============================================================================
# This module generates human-readable explanations for claim verdicts.
#
# Explainability is critical for:
# 1. TRUST: Reviewers need to understand why a verdict was reached
# 2. GOVERNANCE: Audit requirements demand transparent decision-making
# 3. COMPLIANCE: Regulated industries require explainable AI
# 4. IMPROVEMENT: Clear explanations help identify system issues
#
# Every verdict comes with a full explanation of:
# - What data was extracted
# - How it compared to market rates
# - Why the threshold was triggered
# - What confidence level applies
# =============================================================================

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from src.config import settings, Verdict

# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# Main Explanation Function
# =============================================================================

def generate_explanation(
    extracted_data: Dict[str, Any],
    market_data: Dict[str, Any],
    score_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a comprehensive explanation for a claim verdict.
    
    This is the main function for creating explanations. It produces:
    - Human-readable summary
    - Field-by-field breakdown
    - Comparison analysis
    - Confidence assessment
    - Audit trail data
    
    Args:
        extracted_data: Fields extracted from the document.
        market_data: Market rate comparison data.
        score_result: Scoring and verdict information.
    
    Returns:
        dict: Complete explanation package:
            - summary: Brief human-readable explanation
            - detailed_explanation: Full explanation text
            - field_breakdown: Analysis of each extracted field
            - comparison_analysis: Rate comparison breakdown
            - confidence_assessment: Data quality analysis
            - audit_trail: Technical details for compliance
            - generated_at: Timestamp
    
    Example:
        >>> explanation = generate_explanation(extracted, market, score)
        >>> print(explanation['summary'])
        "FLAGGED: Daily rate of £89.00 exceeds market by 37%..."
    """
    verdict = score_result.get("verdict", Verdict.INSUFFICIENT_DATA)
    
    # Generate each component of the explanation
    summary = _generate_summary(extracted_data, market_data, score_result)
    detailed = _generate_detailed_explanation(extracted_data, market_data, score_result)
    field_breakdown = _generate_field_breakdown(extracted_data)
    comparison_analysis = _generate_comparison_analysis(extracted_data, market_data, score_result)
    confidence_assessment = _generate_confidence_assessment(extracted_data, score_result)
    audit_trail = _generate_audit_trail(extracted_data, market_data, score_result)
    
    explanation = {
        "summary": summary,
        "detailed_explanation": detailed,
        "field_breakdown": field_breakdown,
        "comparison_analysis": comparison_analysis,
        "confidence_assessment": confidence_assessment,
        "audit_trail": audit_trail,
        "verdict": verdict,
        "generated_at": datetime.utcnow().isoformat()
    }
    
    logger.info(f"Generated explanation for {verdict} verdict")
    
    return explanation


# =============================================================================
# Summary Generation
# =============================================================================

def _generate_summary(
    extracted_data: Dict[str, Any],
    market_data: Dict[str, Any],
    score_result: Dict[str, Any]
) -> str:
    """
    Generate a brief, one-paragraph summary of the verdict.
    
    This is the "headline" explanation that appears at the top
    of the report. It should be immediately understandable.
    
    Args:
        extracted_data: Extracted claim fields.
        market_data: Market rate data.
        score_result: Scoring result.
    
    Returns:
        str: Brief summary text (2-3 sentences).
    """
    verdict = score_result.get("verdict", Verdict.INSUFFICIENT_DATA)
    daily_rate = extracted_data.get("daily_rate")
    inflation_ratio = score_result.get("inflation_ratio")
    confidence = score_result.get("confidence_score", 0)
    
    market_high = market_data.get("market_rate_high")
    vehicle_class = extracted_data.get("vehicle_class", "Unknown")
    region = extracted_data.get("region", "Unknown region")
    
    # Build summary based on verdict
    if verdict == Verdict.FAIR:
        summary = (
            f"FAIR: The claimed daily rate of £{daily_rate:.2f} for a {vehicle_class} "
            f"vehicle is within the expected market range for {region}. "
            f"No concerns identified. Confidence: {confidence:.0%}."
        )
        
    elif verdict == Verdict.POTENTIALLY_INFLATED:
        excess_pct = (inflation_ratio - 1) * 100 if inflation_ratio else 0
        summary = (
            f"POTENTIALLY INFLATED: The claimed daily rate of £{daily_rate:.2f} "
            f"exceeds the market rate of £{market_high:.2f} by {excess_pct:.0f}%. "
            f"This warrants closer review. Confidence: {confidence:.0%}."
        )
        
    elif verdict == Verdict.FLAGGED:
        excess_pct = (inflation_ratio - 1) * 100 if inflation_ratio else 0
        summary = (
            f"FLAGGED FOR REVIEW: The claimed daily rate of £{daily_rate:.2f} "
            f"significantly exceeds the market rate of £{market_high:.2f} by {excess_pct:.0f}%. "
            f"Priority review recommended. Confidence: {confidence:.0%}."
        )
        
    else:  # INSUFFICIENT_DATA
        summary = (
            f"INSUFFICIENT DATA: Unable to complete automated analysis. "
            f"Key fields could not be extracted from the document. "
            f"Manual review required. Extraction confidence: {confidence:.0%}."
        )
    
    return summary


# =============================================================================
# Detailed Explanation
# =============================================================================

def _generate_detailed_explanation(
    extracted_data: Dict[str, Any],
    market_data: Dict[str, Any],
    score_result: Dict[str, Any]
) -> str:
    """
    Generate a full, detailed explanation of the verdict.
    
    This provides the complete reasoning, suitable for formal
    reports and audit documentation.
    
    Args:
        extracted_data: Extracted claim fields.
        market_data: Market rate data.
        score_result: Scoring result.
    
    Returns:
        str: Detailed explanation text (multiple paragraphs).
    """
    verdict = score_result.get("verdict")
    paragraphs = []
    
    # Paragraph 1: What was analyzed
    daily_rate = extracted_data.get("daily_rate")
    total_claimed = extracted_data.get("total_claimed")
    hire_days = extracted_data.get("hire_days")
    vehicle_class = extracted_data.get("vehicle_class", "Unknown")
    hire_company = extracted_data.get("hire_company", "Unknown company")
    
    para1 = (
        f"This analysis reviewed a credit hire invoice from {hire_company}. "
        f"The claim is for a {vehicle_class} category vehicle "
    )
    
    if hire_days and total_claimed:
        para1 += (
            f"hired for {hire_days} days at £{daily_rate:.2f} per day, "
            f"totalling £{total_claimed:.2f}."
        )
    elif daily_rate:
        para1 += f"at a daily rate of £{daily_rate:.2f}."
    else:
        para1 += "(rate details could not be fully extracted)."
    
    paragraphs.append(para1)
    
    # Paragraph 2: Market comparison
    market_low = market_data.get("market_rate_low")
    market_high = market_data.get("market_rate_high")
    region = extracted_data.get("region", "UK average")
    source = market_data.get("source", "reference database")
    source_year = market_data.get("source_year", "current")
    
    if market_low and market_high:
        para2 = (
            f"The claimed rate was compared against market benchmarks for "
            f"{vehicle_class} vehicles in {region}. According to our {source} "
            f"({source_year} data), the expected market rate range is "
            f"£{market_low:.2f} to £{market_high:.2f} per day."
        )
        paragraphs.append(para2)
    
    # Paragraph 3: Verdict reasoning
    inflation_ratio = score_result.get("inflation_ratio")
    
    if verdict == Verdict.FAIR:
        if daily_rate and market_high and daily_rate <= market_high:
            para3 = (
                f"The claimed rate falls within the expected market range. "
                f"With an inflation ratio of {inflation_ratio:.2f} (below the {settings.THRESHOLD_FAIR} "
                f"threshold), this claim is classified as FAIR. No rate inflation is indicated."
            )
        else:
            para3 = (
                f"Based on the analysis, this claim is classified as FAIR. "
                f"The rate appears consistent with market norms."
            )
        paragraphs.append(para3)
        
    elif verdict == Verdict.POTENTIALLY_INFLATED:
        excess = (inflation_ratio - 1) * 100 if inflation_ratio else 0
        para3 = (
            f"The claimed rate exceeds the market upper bound by {excess:.1f}%. "
            f"With an inflation ratio of {inflation_ratio:.2f} (between "
            f"{settings.THRESHOLD_FAIR} and {settings.THRESHOLD_INFLATED}), this claim is "
            f"classified as POTENTIALLY INFLATED. While not extreme, this warrants "
            f"closer examination to determine if the premium is justified."
        )
        paragraphs.append(para3)
        
    elif verdict == Verdict.FLAGGED:
        excess = (inflation_ratio - 1) * 100 if inflation_ratio else 0
        excess_daily = score_result.get("excess_daily", 0)
        excess_total = score_result.get("excess_total")
        
        para3 = (
            f"The claimed rate significantly exceeds market norms by {excess:.1f}%. "
            f"With an inflation ratio of {inflation_ratio:.2f} (above the "
            f"{settings.THRESHOLD_INFLATED} threshold), this claim is FLAGGED for priority review. "
            f"The daily excess is £{excess_daily:.2f}."
        )
        
        if excess_total and hire_days:
            para3 += f" Over {hire_days} days, this represents £{excess_total:.2f} above market rates."
        
        paragraphs.append(para3)
        
    else:  # INSUFFICIENT_DATA
        para3 = (
            "Due to incomplete data extraction, automated scoring could not be completed. "
            "This may be due to document quality issues, unusual formatting, or "
            "missing required information in the source document. Manual review is required."
        )
        paragraphs.append(para3)
    
    # Paragraph 4: Confidence note
    confidence = score_result.get("confidence_score", 0)
    
    if confidence >= 0.9:
        para4 = (
            f"Confidence in this analysis is HIGH ({confidence:.0%}). "
            f"All key fields were successfully extracted and verified."
        )
    elif confidence >= 0.7:
        para4 = (
            f"Confidence in this analysis is MODERATE ({confidence:.0%}). "
            f"Most fields were extracted successfully. Minor verification recommended."
        )
    else:
        para4 = (
            f"Confidence in this analysis is LOW ({confidence:.0%}). "
            f"Several fields could not be extracted or verified. "
            f"Manual validation of extracted values is strongly recommended."
        )
    
    paragraphs.append(para4)
    
    return "\n\n".join(paragraphs)


# =============================================================================
# Field Breakdown
# =============================================================================

def _generate_field_breakdown(
    extracted_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Generate a field-by-field breakdown of extracted data.
    
    Shows each field, its value, and extraction status.
    
    Args:
        extracted_data: Extracted claim fields.
    
    Returns:
        list: Field breakdown entries.
    """
    fields = [
        {
            "field": "Daily Rate",
            "key": "daily_rate",
            "format": "currency",
            "required": True
        },
        {
            "field": "Total Claimed",
            "key": "total_claimed",
            "format": "currency",
            "required": True
        },
        {
            "field": "Hire Days",
            "key": "hire_days",
            "format": "number",
            "required": True
        },
        {
            "field": "Vehicle Class",
            "key": "vehicle_class",
            "format": "text",
            "required": True
        },
        {
            "field": "Region",
            "key": "region",
            "format": "text",
            "required": False
        },
        {
            "field": "Hire Company",
            "key": "hire_company",
            "format": "text",
            "required": False
        },
        {
            "field": "Hire Start Date",
            "key": "hire_start_date",
            "format": "date",
            "required": True
        },
        {
            "field": "Hire End Date",
            "key": "hire_end_date",
            "format": "date",
            "required": True
        },
        {
            "field": "Claimant Name",
            "key": "claimant_name",
            "format": "text",
            "required": False
        },
        {
            "field": "Vehicle Registration",
            "key": "vehicle_registration",
            "format": "text",
            "required": False
        }
    ]
    
    breakdown = []
    
    for field_def in fields:
        key = field_def["key"]
        value = extracted_data.get(key)
        
        # Format the value for display
        if value is not None:
            if field_def["format"] == "currency":
                display_value = f"£{value:.2f}"
            elif field_def["format"] == "number":
                display_value = str(value)
            else:
                display_value = str(value)
            status = "extracted"
        else:
            display_value = "Not found"
            status = "missing"
        
        breakdown.append({
            "field_name": field_def["field"],
            "value": display_value,
            "raw_value": value,
            "status": status,
            "required": field_def["required"],
            "format": field_def["format"]
        })
    
    return breakdown


# =============================================================================
# Comparison Analysis
# =============================================================================

def _generate_comparison_analysis(
    extracted_data: Dict[str, Any],
    market_data: Dict[str, Any],
    score_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate detailed rate comparison analysis.
    
    Shows exactly how the claimed rate compares to market benchmarks.
    
    Args:
        extracted_data: Extracted claim fields.
        market_data: Market rate data.
        score_result: Scoring result.
    
    Returns:
        dict: Comparison analysis details.
    """
    daily_rate = extracted_data.get("daily_rate", 0)
    market_low = market_data.get("market_rate_low", 0)
    market_high = market_data.get("market_rate_high", 0)
    inflation_ratio = score_result.get("inflation_ratio", 0)
    
    # Determine rate position
    if daily_rate <= market_low:
        position = "below_range"
        position_description = "Below market minimum"
    elif daily_rate <= market_high:
        position = "within_range"
        position_description = "Within market range"
    else:
        position = "above_range"
        position_description = "Above market maximum"
    
    # Calculate metrics
    market_midpoint = (market_low + market_high) / 2 if market_low and market_high else 0
    deviation_from_midpoint = daily_rate - market_midpoint if market_midpoint else 0
    deviation_percentage = (deviation_from_midpoint / market_midpoint * 100) if market_midpoint else 0
    
    analysis = {
        "claimed_rate": daily_rate,
        "market_rate_low": market_low,
        "market_rate_high": market_high,
        "market_midpoint": round(market_midpoint, 2),
        "inflation_ratio": inflation_ratio,
        "position": position,
        "position_description": position_description,
        "deviation_from_midpoint": round(deviation_from_midpoint, 2),
        "deviation_percentage": round(deviation_percentage, 1),
        "thresholds": {
            "fair_threshold": settings.THRESHOLD_FAIR,
            "inflated_threshold": settings.THRESHOLD_INFLATED,
            "claimed_vs_fair": "below" if inflation_ratio < settings.THRESHOLD_FAIR else "above",
            "claimed_vs_inflated": "below" if inflation_ratio < settings.THRESHOLD_INFLATED else "above"
        },
        "market_data_source": market_data.get("source", "unknown"),
        "market_data_year": market_data.get("source_year")
    }
    
    return analysis


# =============================================================================
# Confidence Assessment
# =============================================================================

def _generate_confidence_assessment(
    extracted_data: Dict[str, Any],
    score_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate assessment of data quality and confidence.
    
    Explains why the confidence score is what it is.
    
    Args:
        extracted_data: Extracted claim fields.
        score_result: Scoring result.
    
    Returns:
        dict: Confidence assessment details.
    """
    confidence = score_result.get("confidence_score", 0)
    
    # Determine confidence level
    if confidence >= 0.9:
        level = "HIGH"
        level_description = "All critical fields extracted successfully"
    elif confidence >= 0.7:
        level = "MODERATE"
        level_description = "Most fields extracted, some verification recommended"
    elif confidence >= 0.5:
        level = "LOW"
        level_description = "Several fields missing or uncertain"
    else:
        level = "VERY_LOW"
        level_description = "Significant data quality issues"
    
    # Check each required field
    required_fields = ["daily_rate", "vehicle_class", "hire_days", "total_claimed"]
    missing_required = [f for f in required_fields if not extracted_data.get(f)]
    
    # Identify any concerns
    concerns = []
    
    if missing_required:
        concerns.append(f"Missing required fields: {', '.join(missing_required)}")
    
    if not extracted_data.get("hire_start_date") and not extracted_data.get("hire_end_date"):
        concerns.append("Hire dates not extracted")
    
    if not extracted_data.get("region"):
        concerns.append("Region not identified (using national averages)")
    
    # Calculate total value impact
    daily_rate = extracted_data.get("daily_rate", 0)
    hire_days = extracted_data.get("hire_days", 0)
    total_claimed = extracted_data.get("total_claimed", 0)
    
    if daily_rate and hire_days and total_claimed:
        calculated_total = daily_rate * hire_days
        variance = abs(total_claimed - calculated_total) / total_claimed if total_claimed else 0
        
        if variance > 0.1:  # More than 10% difference
            concerns.append(f"Total claimed differs from calculated by {variance:.1%}")
    
    assessment = {
        "confidence_score": confidence,
        "confidence_level": level,
        "level_description": level_description,
        "missing_required_fields": missing_required,
        "concerns": concerns,
        "recommendation": _get_confidence_recommendation(level),
        "fields_extracted_count": sum(1 for v in extracted_data.values() if v is not None),
        "fields_total_count": len(extracted_data)
    }
    
    return assessment


def _get_confidence_recommendation(level: str) -> str:
    """
    Get recommendation based on confidence level.
    
    Args:
        level: Confidence level (HIGH, MODERATE, LOW, VERY_LOW).
    
    Returns:
        str: Recommendation text.
    """
    recommendations = {
        "HIGH": "Proceed with automated verdict. No additional verification needed.",
        "MODERATE": "Review extracted values for accuracy before finalizing.",
        "LOW": "Manual verification strongly recommended. Check source document.",
        "VERY_LOW": "Do not rely on automated analysis. Full manual review required."
    }
    
    return recommendations.get(level, "Review required.")


# =============================================================================
# Audit Trail
# =============================================================================

def _generate_audit_trail(
    extracted_data: Dict[str, Any],
    market_data: Dict[str, Any],
    score_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate audit trail data for compliance.
    
    This provides the technical details needed for:
    - Regulatory compliance
    - Decision auditing
    - System debugging
    - Legal documentation
    
    Args:
        extracted_data: Extracted claim fields.
        market_data: Market rate data.
        score_result: Scoring result.
    
    Returns:
        dict: Audit trail data.
    """
    audit_trail = {
        "analysis_timestamp": datetime.utcnow().isoformat(),
        "analysis_version": "1.0.0",
        
        # Input data
        "input": {
            "extracted_fields": extracted_data,
            "market_data_source": market_data.get("source"),
            "market_data_year": market_data.get("source_year")
        },
        
        # Processing details
        "processing": {
            "extraction_confidence": score_result.get("confidence_score"),
            "inflation_ratio_calculated": score_result.get("inflation_ratio"),
            "thresholds_applied": {
                "fair": settings.THRESHOLD_FAIR,
                "inflated": settings.THRESHOLD_INFLATED
            }
        },
        
        # Output
        "output": {
            "verdict": score_result.get("verdict"),
            "confidence_score": score_result.get("confidence_score")
        },
        
        # Decision path
        "decision_path": _generate_decision_path(score_result),
        
        # Scoring factors (from scorer)
        "scoring_factors": score_result.get("scoring_factors", [])
    }
    
    return audit_trail


def _generate_decision_path(score_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate step-by-step decision path.
    
    Shows exactly how the verdict was reached.
    
    Args:
        score_result: Scoring result.
    
    Returns:
        list: Decision steps.
    """
    inflation_ratio = score_result.get("inflation_ratio")
    verdict = score_result.get("verdict")
    
    steps = []
    
    # Step 1: Calculate inflation ratio
    steps.append({
        "step": 1,
        "action": "Calculate inflation ratio",
        "description": f"Claimed rate / Market rate high = {inflation_ratio:.3f}" if inflation_ratio else "Unable to calculate (missing data)",
        "result": inflation_ratio
    })
    
    # Step 2: Compare to FAIR threshold
    if inflation_ratio:
        fair_result = "PASS" if inflation_ratio < settings.THRESHOLD_FAIR else "FAIL"
        steps.append({
            "step": 2,
            "action": f"Compare to FAIR threshold ({settings.THRESHOLD_FAIR})",
            "description": f"{inflation_ratio:.3f} {'<' if inflation_ratio < settings.THRESHOLD_FAIR else '>='} {settings.THRESHOLD_FAIR}",
            "result": fair_result
        })
    
    # Step 3: Compare to INFLATED threshold (if needed)
    if inflation_ratio and inflation_ratio >= settings.THRESHOLD_FAIR:
        inflated_result = "PASS" if inflation_ratio < settings.THRESHOLD_INFLATED else "FAIL"
        steps.append({
            "step": 3,
            "action": f"Compare to INFLATED threshold ({settings.THRESHOLD_INFLATED})",
            "description": f"{inflation_ratio:.3f} {'<' if inflation_ratio < settings.THRESHOLD_INFLATED else '>='} {settings.THRESHOLD_INFLATED}",
            "result": inflated_result
        })
    
    # Step 4: Determine verdict
    steps.append({
        "step": len(steps) + 1,
        "action": "Determine verdict",
        "description": f"Based on threshold comparisons",
        "result": verdict
    })
    
    return steps


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(
    extracted_data: Dict[str, Any],
    market_data: Dict[str, Any],
    score_result: Dict[str, Any],
    format: str = "text"
) -> str:
    """
    Generate a formatted report for the claim analysis.
    
    Creates a complete, formatted report suitable for
    printing, emailing, or displaying in the UI.
    
    Args:
        extracted_data: Extracted claim fields.
        market_data: Market rate data.
        score_result: Scoring result.
        format: Output format ("text" or "html").
    
    Returns:
        str: Formatted report.
    """
    explanation = generate_explanation(extracted_data, market_data, score_result)
    
    if format == "html":
        return _generate_html_report(explanation, extracted_data, market_data, score_result)
    else:
        return _generate_text_report(explanation, extracted_data, market_data, score_result)


def _generate_text_report(
    explanation: Dict[str, Any],
    extracted_data: Dict[str, Any],
    market_data: Dict[str, Any],
    score_result: Dict[str, Any]
) -> str:
    """
    Generate plain text report.
    
    Args:
        explanation: Generated explanation.
        extracted_data: Extracted claim fields.
        market_data: Market rate data.
        score_result: Scoring result.
    
    Returns:
        str: Plain text report.
    """
    lines = [
        "=" * 60,
        "CLAIMSNOW - CLAIM ANALYSIS REPORT",
        "=" * 60,
        "",
        f"Generated: {explanation['generated_at']}",
        f"Verdict: {explanation['verdict']}",
        "",
        "-" * 60,
        "SUMMARY",
        "-" * 60,
        explanation['summary'],
        "",
        "-" * 60,
        "EXTRACTED DATA",
        "-" * 60,
    ]
    
    for field in explanation['field_breakdown']:
        status_indicator = "✓" if field['status'] == 'extracted' else "✗"
        required_indicator = "*" if field['required'] else ""
        lines.append(f"  {status_indicator} {field['field_name']}{required_indicator}: {field['value']}")
    
    lines.extend([
        "",
        "-" * 60,
        "MARKET COMPARISON",
        "-" * 60,
    ])
    
    comp = explanation['comparison_analysis']
    lines.append(f"  Claimed Rate: £{comp['claimed_rate']:.2f}")
    lines.append(f"  Market Range: £{comp['market_rate_low']:.2f} - £{comp['market_rate_high']:.2f}")
    lines.append(f"  Inflation Ratio: {comp['inflation_ratio']:.3f}")
    lines.append(f"  Position: {comp['position_description']}")
    
    lines.extend([
        "",
        "-" * 60,
        "DETAILED EXPLANATION",
        "-" * 60,
        explanation['detailed_explanation'],
        "",
        "-" * 60,
        "RECOMMENDATIONS",
        "-" * 60,
    ])
    
    for rec in score_result.get('recommendations', []):
        lines.append(f"  • {rec}")
    
    lines.extend([
        "",
        "=" * 60,
        "END OF REPORT",
        "=" * 60
    ])
    
    return "\n".join(lines)


def _generate_html_report(
    explanation: Dict[str, Any],
    extracted_data: Dict[str, Any],
    market_data: Dict[str, Any],
    score_result: Dict[str, Any]
) -> str:
    """
    Generate HTML report.
    
    Args:
        explanation: Generated explanation.
        extracted_data: Extracted claim fields.
        market_data: Market rate data.
        score_result: Scoring result.
    
    Returns:
        str: HTML report.
    """
    verdict = explanation['verdict']
    
    # Verdict colors
    verdict_colors = {
        Verdict.FAIR: "#22c55e",  # Green
        Verdict.POTENTIALLY_INFLATED: "#f59e0b",  # Yellow
        Verdict.FLAGGED: "#ef4444",  # Red
        Verdict.INSUFFICIENT_DATA: "#6b7280"  # Gray
    }
    
    verdict_color = verdict_colors.get(verdict, "#6b7280")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
            .header {{ background: #1e3a5f; color: white; padding: 20px; border-radius: 8px; }}
            .verdict-badge {{ display: inline-block; padding: 8px 16px; border-radius: 4px; 
                            background: {verdict_color}; color: white; font-weight: bold; }}
            .section {{ margin: 20px 0; padding: 15px; background: #f9fafb; border-radius: 8px; }}
            .section-title {{ font-weight: bold; margin-bottom: 10px; color: #1e3a5f; }}
            table {{ width: 100%; border-collapse: collapse; }}
            td, th {{ padding: 8px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
            .extracted {{ color: #22c55e; }}
            .missing {{ color: #ef4444; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>ClaimsNOW - Analysis Report</h1>
            <p>Generated: {explanation['generated_at']}</p>
            <span class="verdict-badge">{verdict}</span>
        </div>
        
        <div class="section">
            <div class="section-title">Summary</div>
            <p>{explanation['summary']}</p>
        </div>
        
        <div class="section">
            <div class="section-title">Extracted Data</div>
            <table>
                <tr><th>Field</th><th>Value</th><th>Status</th></tr>
    """
    
    for field in explanation['field_breakdown']:
        status_class = "extracted" if field['status'] == 'extracted' else "missing"
        html += f"""
                <tr>
                    <td>{field['field_name']}{'*' if field['required'] else ''}</td>
                    <td>{field['value']}</td>
                    <td class="{status_class}">{field['status'].upper()}</td>
                </tr>
        """
    
    comp = explanation['comparison_analysis']
    html += f"""
            </table>
        </div>
        
        <div class="section">
            <div class="section-title">Market Comparison</div>
            <table>
                <tr><td>Claimed Rate</td><td>£{comp['claimed_rate']:.2f}</td></tr>
                <tr><td>Market Range</td><td>£{comp['market_rate_low']:.2f} - £{comp['market_rate_high']:.2f}</td></tr>
                <tr><td>Inflation Ratio</td><td>{comp['inflation_ratio']:.3f}</td></tr>
                <tr><td>Position</td><td>{comp['position_description']}</td></tr>
            </table>
        </div>
        
        <div class="section">
            <div class="section-title">Detailed Explanation</div>
            <p>{explanation['detailed_explanation'].replace(chr(10)+chr(10), '</p><p>')}</p>
        </div>
        
        <div class="section">
            <div class="section-title">Recommendations</div>
            <ul>
    """
    
    for rec in score_result.get('recommendations', []):
        html += f"<li>{rec}</li>"
    
    html += """
            </ul>
        </div>
    </body>
    </html>
    """
    
    return html
