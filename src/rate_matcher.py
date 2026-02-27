# =============================================================================
# ClaimsNOW - Rate Matching Engine Module
# =============================================================================
# This module compares extracted claim data against market rate benchmarks.
#
# The rate matcher is the core business logic of ClaimsNOW:
# 1. Takes extracted claim data (vehicle class, region, days)
# 2. Queries the reference database for comparable market rates
# 3. Returns the market rate range for comparison
#
# This enables the scorer to determine if a claim is fair or inflated.
# =============================================================================

import logging
from typing import Dict, Optional, Tuple, Any, List

from src.config import settings, VehicleClass, Region
from src.aws_dynamodb import get_market_rate, query_rates_by_vehicle_class

# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# Fallback Rate Data
# =============================================================================

# Fallback rates used when database is unavailable
# These are UK national averages for 2026 (£/day)
# In production, always prefer DynamoDB data
FALLBACK_RATES = {
    VehicleClass.GROUP_A: {"low": 32.00, "high": 45.00},
    VehicleClass.GROUP_B: {"low": 37.00, "high": 50.00},
    VehicleClass.GROUP_C: {"low": 42.00, "high": 58.00},
    VehicleClass.GROUP_D: {"low": 47.00, "high": 68.00},
    VehicleClass.GROUP_E: {"low": 50.00, "high": 76.00},
    VehicleClass.GROUP_F: {"low": 58.00, "high": 90.00},
    VehicleClass.GROUP_G: {"low": 76.00, "high": 118.00},
    VehicleClass.GROUP_H: {"low": 100.00, "high": 155.00},
    VehicleClass.GROUP_I: {"low": 120.00, "high": 180.00},
    VehicleClass.GROUP_J: {"low": 150.00, "high": 220.00},
    VehicleClass.SUV: {"low": 68.00, "high": 108.00},
    VehicleClass.MPV: {"low": 55.00, "high": 86.00},
    VehicleClass.VAN: {"low": 50.00, "high": 80.00},
    VehicleClass.UNKNOWN: {"low": 45.00, "high": 70.00},
}


# =============================================================================
# Main Rate Matching Function
# =============================================================================

def find_market_rate(
    vehicle_class: str,
    region: str = None,
    hire_days: int = None
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Find the market rate benchmark for a given claim profile.
    
    This is the main function for rate matching. It:
    1. Queries the DynamoDB reference database
    2. Falls back to national averages if specific data unavailable
    3. Returns the market rate range for comparison
    
    Args:
        vehicle_class: The vehicle category (e.g., "GROUP_C", "SUV")
        region: UK region (e.g., "LONDON", "NORTH_WEST"). Optional.
        hire_days: Number of hire days. Optional (for period-specific rates).
    
    Returns:
        Tuple containing:
        - success (bool): True if a rate was found
        - rate_data (dict|None): Market rate information:
            - vehicle_class: The vehicle class matched
            - region: The region matched (may differ from input if fallback used)
            - market_rate_low: Lower bound of fair rate (£/day)
            - market_rate_high: Upper bound of fair rate (£/day)
            - source: Where the rate came from ("database" or "fallback")
            - source_year: Year the rate data applies to
            - hire_period_min: Minimum hire days for this rate
            - hire_period_max: Maximum hire days for this rate
        - error (str|None): Error message if failed
    
    Example:
        >>> success, rate, error = find_market_rate("GROUP_C", "LONDON", 14)
        >>> if success:
        ...     print(f"Market rate: £{rate['market_rate_low']}-£{rate['market_rate_high']}/day")
    """
    # Normalize inputs
    vehicle_class = _normalize_vehicle_class(vehicle_class)
    region = _normalize_region(region)
    
    logger.info(f"Finding market rate for {vehicle_class} in {region}")
    
    # Try to get rate from database
    success, db_rate, error = get_market_rate(vehicle_class, region, hire_days or 1)
    
    if success and db_rate:
        # Found rate in database
        rate_data = {
            **db_rate,
            "source": "database"
        }
        logger.info(f"Found database rate: £{rate_data['market_rate_low']}-£{rate_data['market_rate_high']}")
        return True, rate_data, None
    
    # Database lookup failed - try fallback strategies
    logger.warning(f"Database rate not found: {error}. Trying fallbacks.")
    
    # Strategy 1: Try with UNKNOWN region
    if region != Region.UNKNOWN:
        success, db_rate, _ = get_market_rate(vehicle_class, Region.UNKNOWN, hire_days or 1)
        if success and db_rate:
            rate_data = {
                **db_rate,
                "source": "database",
                "note": f"Used national average (region {region} not in database)"
            }
            return True, rate_data, None
    
    # Strategy 2: Try with UNKNOWN vehicle class
    if vehicle_class != VehicleClass.UNKNOWN:
        success, db_rate, _ = get_market_rate(VehicleClass.UNKNOWN, region, hire_days or 1)
        if success and db_rate:
            rate_data = {
                **db_rate,
                "source": "database",
                "note": f"Used default vehicle class (class {vehicle_class} not in database)"
            }
            return True, rate_data, None
    
    # Strategy 3: Use hardcoded fallback rates
    fallback = _get_fallback_rate(vehicle_class)
    
    rate_data = {
        "vehicle_class": vehicle_class,
        "region": region or Region.UNKNOWN,
        "market_rate_low": fallback["low"],
        "market_rate_high": fallback["high"],
        "source": "fallback",
        "source_year": 2026,
        "hire_period_min": 1,
        "hire_period_max": 365,
        "note": "Used fallback rates (database unavailable)"
    }
    
    logger.warning(f"Using fallback rate: £{rate_data['market_rate_low']}-£{rate_data['market_rate_high']}")
    return True, rate_data, None


def _normalize_vehicle_class(vehicle_class: str) -> str:
    """
    Normalize vehicle class to standard format.
    
    Handles variations like:
    - "Group C" -> "GROUP_C"
    - "C" -> "GROUP_C"
    - "group_c" -> "GROUP_C"
    
    Args:
        vehicle_class: Raw vehicle class string.
    
    Returns:
        str: Normalized vehicle class constant.
    """
    if not vehicle_class:
        return VehicleClass.UNKNOWN
    
    vc = vehicle_class.upper().strip()
    
    # If already in correct format
    if hasattr(VehicleClass, vc.replace(" ", "_")):
        return vc.replace(" ", "_")
    
    # Handle "Group X" format
    if vc.startswith("GROUP"):
        return vc.replace(" ", "_")
    
    # Handle single letter
    if len(vc) == 1 and vc in "ABCDEFGHIJ":
        return f"GROUP_{vc}"
    
    # Handle special types
    if "SUV" in vc:
        return VehicleClass.SUV
    if "MPV" in vc:
        return VehicleClass.MPV
    if "VAN" in vc:
        return VehicleClass.VAN
    
    return VehicleClass.UNKNOWN


def _normalize_region(region: str) -> str:
    """
    Normalize region to standard format.
    
    Args:
        region: Raw region string.
    
    Returns:
        str: Normalized region constant.
    """
    if not region:
        return Region.UNKNOWN
    
    r = region.upper().strip().replace(" ", "_")
    
    # Check if it's a valid region
    if hasattr(Region, r):
        return r
    
    return Region.UNKNOWN


def _get_fallback_rate(vehicle_class: str) -> Dict[str, float]:
    """
    Get fallback rate for a vehicle class.
    
    Args:
        vehicle_class: Normalized vehicle class.
    
    Returns:
        dict: {"low": float, "high": float} rate range.
    """
    return FALLBACK_RATES.get(
        vehicle_class,
        FALLBACK_RATES[VehicleClass.UNKNOWN]
    )


# =============================================================================
# Rate Comparison Functions
# =============================================================================

def compare_rate(
    claimed_rate: float,
    market_rate_low: float,
    market_rate_high: float
) -> Dict[str, Any]:
    """
    Compare a claimed rate against market benchmarks.
    
    Calculates various metrics for understanding how the claimed
    rate relates to the market:
    - Whether it falls within the market range
    - How much it exceeds the upper bound (if any)
    - The inflation ratio
    
    Args:
        claimed_rate: The daily rate being claimed (£).
        market_rate_low: Lower bound of market rate (£).
        market_rate_high: Upper bound of market rate (£).
    
    Returns:
        dict: Comparison metrics:
            - within_range: True if claimed_rate is between low and high
            - below_range: True if claimed_rate is below low
            - above_range: True if claimed_rate is above high
            - inflation_ratio: claimed_rate / market_rate_high
            - excess_amount: How much above high (or 0)
            - excess_percentage: Percentage above high (or 0)
            - market_midpoint: Average of low and high
    
    Example:
        >>> comparison = compare_rate(89.00, 45.00, 65.00)
        >>> print(f"Inflation ratio: {comparison['inflation_ratio']:.2f}")
        >>> print(f"Above market by: £{comparison['excess_amount']:.2f}")
    """
    # Calculate the midpoint for reference
    market_midpoint = (market_rate_low + market_rate_high) / 2
    
    # Determine range position
    within_range = market_rate_low <= claimed_rate <= market_rate_high
    below_range = claimed_rate < market_rate_low
    above_range = claimed_rate > market_rate_high
    
    # Calculate inflation ratio (vs upper bound)
    # This is the key metric for scoring
    inflation_ratio = claimed_rate / market_rate_high if market_rate_high > 0 else 0
    
    # Calculate excess if above range
    if above_range:
        excess_amount = claimed_rate - market_rate_high
        excess_percentage = (excess_amount / market_rate_high) * 100
    else:
        excess_amount = 0
        excess_percentage = 0
    
    comparison = {
        "within_range": within_range,
        "below_range": below_range,
        "above_range": above_range,
        "inflation_ratio": round(inflation_ratio, 3),
        "excess_amount": round(excess_amount, 2),
        "excess_percentage": round(excess_percentage, 1),
        "market_midpoint": round(market_midpoint, 2),
        "claimed_rate": claimed_rate,
        "market_rate_low": market_rate_low,
        "market_rate_high": market_rate_high
    }
    
    logger.debug(f"Rate comparison: claimed £{claimed_rate} vs market £{market_rate_low}-£{market_rate_high}, ratio={inflation_ratio:.2f}")
    
    return comparison


def get_regional_comparison(
    vehicle_class: str,
    claimed_rate: float
) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
    """
    Compare claimed rate against all regions for a vehicle class.
    
    Useful for showing how the claim compares across different
    UK regions, which can provide context for borderline cases.
    
    Args:
        vehicle_class: The vehicle class to compare.
        claimed_rate: The daily rate being claimed.
    
    Returns:
        Tuple containing:
        - success (bool): True if comparison succeeded
        - comparisons (list): Regional comparison data
        - error (str|None): Error message if failed
    
    Example:
        >>> success, comparisons, _ = get_regional_comparison("GROUP_C", 89.00)
        >>> for comp in comparisons:
        ...     print(f"{comp['region']}: {comp['position']}")
    """
    vehicle_class = _normalize_vehicle_class(vehicle_class)
    
    # Get rates for all regions
    success, rates, error = query_rates_by_vehicle_class(vehicle_class)
    
    if not success:
        return False, [], error
    
    comparisons = []
    
    for rate in rates:
        comparison = compare_rate(
            claimed_rate,
            rate["market_rate_low"],
            rate["market_rate_high"]
        )
        
        # Determine position label
        if comparison["within_range"]:
            position = "WITHIN_RANGE"
        elif comparison["below_range"]:
            position = "BELOW_RANGE"
        elif comparison["inflation_ratio"] < settings.THRESHOLD_FAIR:
            position = "SLIGHTLY_ABOVE"
        elif comparison["inflation_ratio"] < settings.THRESHOLD_INFLATED:
            position = "MODERATELY_ABOVE"
        else:
            position = "SIGNIFICANTLY_ABOVE"
        
        comparisons.append({
            "region": rate["region"],
            "market_rate_low": rate["market_rate_low"],
            "market_rate_high": rate["market_rate_high"],
            "inflation_ratio": comparison["inflation_ratio"],
            "position": position
        })
    
    # Sort by inflation ratio (lowest first)
    comparisons.sort(key=lambda x: x["inflation_ratio"])
    
    return True, comparisons, None


# =============================================================================
# Period Adjustment Functions
# =============================================================================

def adjust_rate_for_period(
    base_rate_low: float,
    base_rate_high: float,
    hire_days: int
) -> Dict[str, float]:
    """
    Adjust market rates based on hire period length.
    
    Longer hire periods typically get discounts:
    - 1-7 days: Standard rate
    - 8-14 days: 5% discount
    - 15-30 days: 10% discount
    - 31+ days: 15% discount
    
    This reflects real-world pricing practices.
    
    Args:
        base_rate_low: Base market rate lower bound.
        base_rate_high: Base market rate upper bound.
        hire_days: Number of hire days.
    
    Returns:
        dict: Adjusted rates {"low": float, "high": float}
    
    Example:
        >>> adjusted = adjust_rate_for_period(50.00, 70.00, 21)
        >>> print(f"21-day rate: £{adjusted['low']}-£{adjusted['high']}")
        # Output: 21-day rate: £45.00-£63.00 (10% discount)
    """
    # Determine discount tier
    if hire_days <= 7:
        discount = 0.0
    elif hire_days <= 14:
        discount = 0.05  # 5% discount
    elif hire_days <= 30:
        discount = 0.10  # 10% discount
    else:
        discount = 0.15  # 15% discount
    
    # Apply discount to upper bound only
    # Lower bound stays the same (represents minimum acceptable rate)
    adjusted_high = base_rate_high * (1 - discount)
    
    return {
        "low": base_rate_low,
        "high": round(adjusted_high, 2),
        "discount_applied": discount,
        "hire_days": hire_days
    }


# =============================================================================
# Validation Functions
# =============================================================================

def validate_rate_data(rate_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate market rate data for reasonableness.
    
    Checks:
    - Rates are positive numbers
    - Low rate is less than high rate
    - Rates are within realistic bounds
    
    Args:
        rate_data: Market rate dictionary to validate.
    
    Returns:
        Tuple containing:
        - valid (bool): True if data passes validation
        - issues (list): List of validation issue messages
    """
    issues = []
    
    low = rate_data.get("market_rate_low", 0)
    high = rate_data.get("market_rate_high", 0)
    
    # Check rates are positive
    if low <= 0:
        issues.append("Market rate low must be positive")
    if high <= 0:
        issues.append("Market rate high must be positive")
    
    # Check low <= high
    if low > high:
        issues.append(f"Market rate low (£{low}) exceeds high (£{high})")
    
    # Check reasonable bounds
    if low > 0 and low < 10:
        issues.append(f"Market rate low (£{low}) seems unrealistically low")
    if high > 500:
        issues.append(f"Market rate high (£{high}) seems unrealistically high")
    
    # Check source year
    source_year = rate_data.get("source_year", 0)
    if source_year and source_year < 2020:
        issues.append(f"Rate data from {source_year} may be outdated")
    
    return len(issues) == 0, issues


# =============================================================================
# Summary Functions
# =============================================================================

def generate_rate_summary(
    claimed_rate: float,
    rate_data: Dict[str, Any],
    comparison: Dict[str, Any]
) -> str:
    """
    Generate a human-readable rate comparison summary.
    
    Creates a text summary suitable for reports and explanations.
    
    Args:
        claimed_rate: The daily rate being claimed.
        rate_data: Market rate information.
        comparison: Rate comparison metrics.
    
    Returns:
        str: Human-readable summary text.
    
    Example:
        >>> summary = generate_rate_summary(89.00, rate_data, comparison)
        >>> print(summary)
        "The claimed daily rate of £89.00 exceeds the market range of 
         £45.00-£65.00 for GROUP_C vehicles in LONDON by 36.9%..."
    """
    vehicle_class = rate_data.get("vehicle_class", "Unknown")
    region = rate_data.get("region", "Unknown")
    low = rate_data.get("market_rate_low", 0)
    high = rate_data.get("market_rate_high", 0)
    
    # Build summary based on comparison result
    if comparison["within_range"]:
        summary = (
            f"The claimed daily rate of £{claimed_rate:.2f} falls within "
            f"the market range of £{low:.2f}-£{high:.2f} for {vehicle_class} "
            f"vehicles in {region}. This rate appears fair."
        )
    elif comparison["below_range"]:
        summary = (
            f"The claimed daily rate of £{claimed_rate:.2f} is below "
            f"the typical market range of £{low:.2f}-£{high:.2f} for {vehicle_class} "
            f"vehicles in {region}. This rate is favourable."
        )
    else:
        excess_pct = comparison["excess_percentage"]
        summary = (
            f"The claimed daily rate of £{claimed_rate:.2f} exceeds "
            f"the market range of £{low:.2f}-£{high:.2f} for {vehicle_class} "
            f"vehicles in {region} by {excess_pct:.1f}%. "
            f"The inflation ratio is {comparison['inflation_ratio']:.2f}."
        )
    
    # Add source note
    source = rate_data.get("source", "unknown")
    source_year = rate_data.get("source_year", "N/A")
    
    summary += f" (Rate source: {source}, {source_year})"
    
    # Add any additional notes
    note = rate_data.get("note")
    if note:
        summary += f" Note: {note}"
    
    return summary
