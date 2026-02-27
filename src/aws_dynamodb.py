# =============================================================================
# ClaimsNOW - AWS DynamoDB Integration Module
# =============================================================================
# This module handles all database operations using Amazon DynamoDB.
#
# DynamoDB is used for two main purposes:
# 1. MARKET RATES TABLE: Stores reference hire rates for comparison
#    - Used to determine if a claimed rate is fair or inflated
#    - Contains vehicle class, region, and rate ranges
#
# 2. CLAIMS TABLE: Stores claim analysis results for audit
#    - Every analysis is logged for compliance
#    - Allows retrieval of historical analyses
# =============================================================================

import uuid
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
import boto3
from botocore.exceptions import ClientError

from src.config import settings, VehicleClass, Region

# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# DynamoDB Client Initialization
# =============================================================================

def get_dynamodb_resource():
    """
    Create and return a DynamoDB resource.
    
    We use the resource (not client) for cleaner syntax when
    working with table items.
    
    Returns:
        boto3.resource: Configured DynamoDB resource.
    """
    resource_config = {
        "service_name": "dynamodb",
        "region_name": settings.AWS_REGION,
    }
    
    if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
        resource_config["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
        resource_config["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
    
    return boto3.resource(**resource_config)


def get_rates_table():
    """
    Get the market rates reference table.
    
    Returns:
        Table: DynamoDB table object for market rates.
    """
    dynamodb = get_dynamodb_resource()
    return dynamodb.Table(settings.DYNAMODB_RATES_TABLE)


def get_claims_table():
    """
    Get the claims history table.
    
    Returns:
        Table: DynamoDB table object for claim analyses.
    """
    dynamodb = get_dynamodb_resource()
    return dynamodb.Table(settings.DYNAMODB_CLAIMS_TABLE)


# =============================================================================
# Market Rate Functions
# =============================================================================

def get_market_rate(
    vehicle_class: str,
    region: str,
    hire_days: int
) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """
    Retrieve market rate benchmark for a specific combination.
    
    The market rate represents the expected fair price range
    for hiring a vehicle of the given class in the given region
    for the specified number of days.
    
    Args:
        vehicle_class: Vehicle group (e.g., "GROUP_C", "SUV")
        region: UK region (e.g., "LONDON", "NORTH_WEST")
        hire_days: Number of hire days
    
    Returns:
        Tuple containing:
        - success (bool): True if rate was found
        - rate (dict|None): Rate data with keys:
            - vehicle_class: The vehicle class
            - region: The region
            - hire_period_min: Minimum days this rate applies to
            - hire_period_max: Maximum days this rate applies to
            - market_rate_low: Lower bound of fair rate (£/day)
            - market_rate_high: Upper bound of fair rate (£/day)
            - source_year: Year the rate data was collected
        - error (str|None): Error message if failed
    
    Example:
        >>> success, rate, error = get_market_rate("GROUP_C", "LONDON", 14)
        >>> if success:
        ...     print(f"Fair rate: £{rate['market_rate_low']}-£{rate['market_rate_high']}/day")
    """
    try:
        table = get_rates_table()
        
        # Query for matching rate
        # Primary key structure: vehicle_class (partition), region (sort)
        # We then filter by hire period
        response = table.get_item(
            Key={
                "vehicle_class": vehicle_class,
                "region": region
            }
        )
        
        item = response.get("Item")
        
        if not item:
            # Try with UNKNOWN region as fallback
            response = table.get_item(
                Key={
                    "vehicle_class": vehicle_class,
                    "region": Region.UNKNOWN
                }
            )
            item = response.get("Item")
        
        if not item:
            error_msg = f"No market rate found for {vehicle_class} in {region}"
            logger.warning(error_msg)
            return False, None, error_msg
        
        # Convert Decimal values to float for JSON serialization
        rate_data = {
            "vehicle_class": item.get("vehicle_class"),
            "region": item.get("region"),
            "hire_period_min": int(item.get("hire_period_min", 1)),
            "hire_period_max": int(item.get("hire_period_max", 365)),
            "market_rate_low": float(item.get("market_rate_low", 0)),
            "market_rate_high": float(item.get("market_rate_high", 0)),
            "source_year": int(item.get("source_year", 2026))
        }
        
        logger.info(f"Found market rate for {vehicle_class}/{region}: £{rate_data['market_rate_low']}-£{rate_data['market_rate_high']}")
        return True, rate_data, None
        
    except ClientError as e:
        error_msg = f"DynamoDB query failed: {str(e)}"
        logger.error(error_msg)
        return False, None, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error getting market rate: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, None, error_msg


def query_rates_by_vehicle_class(
    vehicle_class: str
) -> Tuple[bool, List[Dict], Optional[str]]:
    """
    Get all market rates for a specific vehicle class.
    
    Useful for showing rate comparison across regions.
    
    Args:
        vehicle_class: Vehicle group to query.
    
    Returns:
        Tuple containing:
        - success (bool): True if query succeeded
        - rates (list): List of rate records for all regions
        - error (str|None): Error message if failed
    """
    try:
        table = get_rates_table()
        
        # Query all items with this partition key
        response = table.query(
            KeyConditionExpression="vehicle_class = :vc",
            ExpressionAttributeValues={
                ":vc": vehicle_class
            }
        )
        
        items = response.get("Items", [])
        
        # Convert Decimal values to float
        rates = []
        for item in items:
            rates.append({
                "vehicle_class": item.get("vehicle_class"),
                "region": item.get("region"),
                "market_rate_low": float(item.get("market_rate_low", 0)),
                "market_rate_high": float(item.get("market_rate_high", 0)),
                "source_year": int(item.get("source_year", 2026))
            })
        
        logger.info(f"Found {len(rates)} rate records for {vehicle_class}")
        return True, rates, None
        
    except ClientError as e:
        error_msg = f"DynamoDB query failed: {str(e)}"
        logger.error(error_msg)
        return False, [], error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, [], error_msg


def put_market_rate(
    vehicle_class: str,
    region: str,
    market_rate_low: float,
    market_rate_high: float,
    hire_period_min: int = 1,
    hire_period_max: int = 365,
    source_year: int = 2026
) -> Tuple[bool, Optional[str]]:
    """
    Insert or update a market rate record.
    
    Used for populating the reference database with benchmark rates.
    
    Args:
        vehicle_class: Vehicle group (e.g., "GROUP_C")
        region: UK region (e.g., "LONDON")
        market_rate_low: Lower bound of fair daily rate (£)
        market_rate_high: Upper bound of fair daily rate (£)
        hire_period_min: Minimum hire days this rate applies to
        hire_period_max: Maximum hire days this rate applies to
        source_year: Year this rate data was collected
    
    Returns:
        Tuple containing:
        - success (bool): True if write succeeded
        - error (str|None): Error message if failed
    
    Example:
        >>> success, error = put_market_rate(
        ...     vehicle_class="GROUP_C",
        ...     region="LONDON",
        ...     market_rate_low=45.00,
        ...     market_rate_high=65.00
        ... )
    """
    try:
        table = get_rates_table()
        
        # DynamoDB requires Decimal for numbers, not float
        table.put_item(
            Item={
                "vehicle_class": vehicle_class,
                "region": region,
                "market_rate_low": Decimal(str(market_rate_low)),
                "market_rate_high": Decimal(str(market_rate_high)),
                "hire_period_min": hire_period_min,
                "hire_period_max": hire_period_max,
                "source_year": source_year,
                "updated_at": datetime.utcnow().isoformat()
            }
        )
        
        logger.info(f"Saved market rate: {vehicle_class}/{region} = £{market_rate_low}-£{market_rate_high}")
        return True, None
        
    except ClientError as e:
        error_msg = f"Failed to save market rate: {str(e)}"
        logger.error(error_msg)
        return False, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg


# =============================================================================
# Claim Analysis Functions
# =============================================================================

def store_claim_result(
    claim_data: Dict[str, Any]
) -> Tuple[bool, str, Optional[str]]:
    """
    Store a claim analysis result for audit trail.
    
    Every analysis is saved to support:
    - Compliance and governance requirements
    - Historical lookup and reporting
    - Audit trail for disputed claims
    
    Args:
        claim_data: The complete analysis result. Should include:
            - extracted_fields: Data extracted from document
            - market_comparison: Rate comparison results
            - verdict: FAIR, POTENTIALLY_INFLATED, or FLAGGED
            - confidence_score: Extraction confidence
            - explanation: Human-readable reasoning
    
    Returns:
        Tuple containing:
        - success (bool): True if save succeeded
        - claim_id (str): Unique ID assigned to this claim
        - error (str|None): Error message if failed
    
    Example:
        >>> result = {"verdict": "FAIR", "confidence": 0.95, ...}
        >>> success, claim_id, error = store_claim_result(result)
        >>> print(f"Claim saved with ID: {claim_id}")
    """
    # Generate unique claim ID
    claim_id = str(uuid.uuid4())
    
    try:
        table = get_claims_table()
        
        # Prepare the item for DynamoDB
        # Convert any float values to Decimal
        item = _convert_floats_to_decimal({
            "claim_id": claim_id,
            "created_at": datetime.utcnow().isoformat(),
            **claim_data
        })
        
        table.put_item(Item=item)
        
        logger.info(f"Stored claim result with ID: {claim_id}")
        return True, claim_id, None
        
    except ClientError as e:
        error_msg = f"Failed to store claim result: {str(e)}"
        logger.error(error_msg)
        return False, "", error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, "", error_msg


def get_claim_result(claim_id: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """
    Retrieve a previously stored claim analysis.
    
    Used for:
    - Looking up historical analyses
    - Fetching results for the frontend
    - Audit and compliance reviews
    
    Args:
        claim_id: The unique ID of the claim to retrieve.
    
    Returns:
        Tuple containing:
        - success (bool): True if claim was found
        - claim (dict|None): The claim data if found
        - error (str|None): Error message if failed or not found
    
    Example:
        >>> success, claim, error = get_claim_result("abc-123-def")
        >>> if success:
        ...     print(f"Verdict: {claim['verdict']}")
    """
    try:
        table = get_claims_table()
        
        response = table.get_item(
            Key={"claim_id": claim_id}
        )
        
        item = response.get("Item")
        
        if not item:
            error_msg = f"Claim not found: {claim_id}"
            logger.warning(error_msg)
            return False, None, error_msg
        
        # Convert Decimal values back to float for JSON serialization
        claim_data = _convert_decimals_to_float(item)
        
        logger.info(f"Retrieved claim: {claim_id}")
        return True, claim_data, None
        
    except ClientError as e:
        error_msg = f"Failed to retrieve claim: {str(e)}"
        logger.error(error_msg)
        return False, None, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, None, error_msg


def list_recent_claims(
    limit: int = 20
) -> Tuple[bool, List[Dict], Optional[str]]:
    """
    List recent claim analyses.
    
    Returns claims sorted by creation time (most recent first).
    Useful for dashboard displays and admin views.
    
    Args:
        limit: Maximum number of claims to return. Default 20.
    
    Returns:
        Tuple containing:
        - success (bool): True if query succeeded
        - claims (list): List of claim summaries
        - error (str|None): Error message if failed
    """
    try:
        table = get_claims_table()
        
        # Scan for recent claims
        # Note: For production, use a GSI with created_at as sort key
        response = table.scan(
            Limit=limit,
            ProjectionExpression="claim_id, created_at, verdict, confidence_score"
        )
        
        items = response.get("Items", [])
        
        # Convert and sort by created_at
        claims = [_convert_decimals_to_float(item) for item in items]
        claims.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        logger.info(f"Listed {len(claims)} recent claims")
        return True, claims, None
        
    except ClientError as e:
        error_msg = f"Failed to list claims: {str(e)}"
        logger.error(error_msg)
        return False, [], error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, [], error_msg


def update_claim_status(
    claim_id: str,
    status: str,
    reviewer_notes: str = None
) -> Tuple[bool, Optional[str]]:
    """
    Update the status of a claim after human review.
    
    Allows reviewers to mark claims as:
    - "reviewed" - Human has checked the analysis
    - "approved" - Claim accepted as-is
    - "disputed" - Claim is under dispute
    - "closed" - Claim processing complete
    
    Args:
        claim_id: The claim to update.
        status: New status value.
        reviewer_notes: Optional notes from the reviewer.
    
    Returns:
        Tuple containing:
        - success (bool): True if update succeeded
        - error (str|None): Error message if failed
    """
    try:
        table = get_claims_table()
        
        # Build update expression
        update_expr = "SET review_status = :status, reviewed_at = :reviewed_at"
        expr_values = {
            ":status": status,
            ":reviewed_at": datetime.utcnow().isoformat()
        }
        
        if reviewer_notes:
            update_expr += ", reviewer_notes = :notes"
            expr_values[":notes"] = reviewer_notes
        
        table.update_item(
            Key={"claim_id": claim_id},
            UpdateExpression=update_expr,
            ExpressionAttributeValues=expr_values
        )
        
        logger.info(f"Updated claim {claim_id} status to: {status}")
        return True, None
        
    except ClientError as e:
        error_msg = f"Failed to update claim: {str(e)}"
        logger.error(error_msg)
        return False, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg


# =============================================================================
# Helper Functions
# =============================================================================

def _convert_floats_to_decimal(obj: Any) -> Any:
    """
    Recursively convert float values to Decimal for DynamoDB.
    
    DynamoDB doesn't accept Python floats directly - they must be
    converted to Decimal for precise numeric storage.
    
    Args:
        obj: The object to convert (dict, list, or scalar).
    
    Returns:
        The converted object with floats as Decimals.
    """
    if isinstance(obj, dict):
        return {k: _convert_floats_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_floats_to_decimal(item) for item in obj]
    elif isinstance(obj, float):
        return Decimal(str(obj))
    else:
        return obj


def _convert_decimals_to_float(obj: Any) -> Any:
    """
    Recursively convert Decimal values to float for JSON serialization.
    
    When reading from DynamoDB, numeric values come back as Decimal.
    These need to be converted to float for JSON responses.
    
    Args:
        obj: The object to convert (dict, list, or scalar).
    
    Returns:
        The converted object with Decimals as floats.
    """
    if isinstance(obj, dict):
        return {k: _convert_decimals_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_decimals_to_float(item) for item in obj]
    elif isinstance(obj, Decimal):
        # Use int if the decimal is whole, otherwise float
        if obj % 1 == 0:
            return int(obj)
        return float(obj)
    else:
        return obj


# =============================================================================
# Database Setup Helpers
# =============================================================================

def seed_sample_rates() -> Tuple[bool, int, Optional[str]]:
    """
    Populate the market rates table with sample data.
    
    This creates a realistic reference database for testing.
    In production, this data would come from Whichrate's 65M+ records.
    
    Returns:
        Tuple containing:
        - success (bool): True if seeding succeeded
        - count (int): Number of records created
        - error (str|None): Error message if failed
    
    Note:
        These are sample rates for demonstration only.
        Real rates would vary by region and be updated regularly.
    """
    # Sample rates by vehicle class (London region, 2026)
    # Rates are in GBP per day
    sample_rates = [
        # Vehicle Class, Region, Low Rate, High Rate
        (VehicleClass.GROUP_A, Region.LONDON, 35.00, 50.00),
        (VehicleClass.GROUP_B, Region.LONDON, 40.00, 55.00),
        (VehicleClass.GROUP_C, Region.LONDON, 45.00, 65.00),
        (VehicleClass.GROUP_D, Region.LONDON, 50.00, 75.00),
        (VehicleClass.GROUP_E, Region.LONDON, 55.00, 85.00),
        (VehicleClass.GROUP_F, Region.LONDON, 65.00, 100.00),
        (VehicleClass.GROUP_G, Region.LONDON, 85.00, 130.00),
        (VehicleClass.GROUP_H, Region.LONDON, 110.00, 170.00),
        (VehicleClass.SUV, Region.LONDON, 75.00, 120.00),
        (VehicleClass.MPV, Region.LONDON, 60.00, 95.00),
        (VehicleClass.VAN, Region.LONDON, 55.00, 90.00),
        
        # North West region (typically 10-15% lower than London)
        (VehicleClass.GROUP_A, Region.NORTH_WEST, 30.00, 42.00),
        (VehicleClass.GROUP_B, Region.NORTH_WEST, 35.00, 48.00),
        (VehicleClass.GROUP_C, Region.NORTH_WEST, 40.00, 55.00),
        (VehicleClass.GROUP_D, Region.NORTH_WEST, 45.00, 65.00),
        (VehicleClass.GROUP_E, Region.NORTH_WEST, 48.00, 72.00),
        (VehicleClass.GROUP_F, Region.NORTH_WEST, 55.00, 85.00),
        (VehicleClass.GROUP_G, Region.NORTH_WEST, 72.00, 110.00),
        (VehicleClass.GROUP_H, Region.NORTH_WEST, 95.00, 145.00),
        (VehicleClass.SUV, Region.NORTH_WEST, 65.00, 100.00),
        (VehicleClass.MPV, Region.NORTH_WEST, 52.00, 80.00),
        (VehicleClass.VAN, Region.NORTH_WEST, 48.00, 75.00),
        
        # Unknown region (national average for fallback)
        (VehicleClass.GROUP_A, Region.UNKNOWN, 32.00, 45.00),
        (VehicleClass.GROUP_B, Region.UNKNOWN, 37.00, 50.00),
        (VehicleClass.GROUP_C, Region.UNKNOWN, 42.00, 58.00),
        (VehicleClass.GROUP_D, Region.UNKNOWN, 47.00, 68.00),
        (VehicleClass.GROUP_E, Region.UNKNOWN, 50.00, 76.00),
        (VehicleClass.GROUP_F, Region.UNKNOWN, 58.00, 90.00),
        (VehicleClass.GROUP_G, Region.UNKNOWN, 76.00, 118.00),
        (VehicleClass.GROUP_H, Region.UNKNOWN, 100.00, 155.00),
        (VehicleClass.SUV, Region.UNKNOWN, 68.00, 108.00),
        (VehicleClass.MPV, Region.UNKNOWN, 55.00, 86.00),
        (VehicleClass.VAN, Region.UNKNOWN, 50.00, 80.00),
        (VehicleClass.UNKNOWN, Region.UNKNOWN, 45.00, 70.00),
    ]
    
    count = 0
    errors = []
    
    for vehicle_class, region, rate_low, rate_high in sample_rates:
        success, error = put_market_rate(
            vehicle_class=vehicle_class,
            region=region,
            market_rate_low=rate_low,
            market_rate_high=rate_high
        )
        
        if success:
            count += 1
        else:
            errors.append(f"{vehicle_class}/{region}: {error}")
    
    if errors:
        return False, count, f"Some records failed: {'; '.join(errors[:5])}"
    
    logger.info(f"Seeded {count} sample market rates")
    return True, count, None
