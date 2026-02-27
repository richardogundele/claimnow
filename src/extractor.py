# =============================================================================
# ClaimsNOW - Field Extraction Module
# =============================================================================
# This module extracts structured data from document text using regex patterns.
#
# The extractor is the "deterministic" first pass of extraction:
# 1. Fast - regex is much faster than AI
# 2. Predictable - same input always gives same output
# 3. Cost-effective - no API calls
#
# When regex extraction fails to find all required fields, the system
# escalates to the Bedrock module for AI-powered extraction.
# =============================================================================

import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dateutil import parser as date_parser

from src.config import settings, VehicleClass, Region

# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# Required Fields Definition
# =============================================================================

# These are the fields we need to extract from every document
# Each field has a name, description, and whether it's required for scoring

REQUIRED_FIELDS = [
    {"name": "hire_start_date", "required": True, "description": "Start of hire period"},
    {"name": "hire_end_date", "required": True, "description": "End of hire period"},
    {"name": "hire_days", "required": True, "description": "Total days of hire"},
    {"name": "vehicle_class", "required": True, "description": "Vehicle category/group"},
    {"name": "daily_rate", "required": True, "description": "Daily hire rate in GBP"},
    {"name": "total_claimed", "required": True, "description": "Total amount claimed"},
    {"name": "hire_company", "required": False, "description": "Name of hire company"},
    {"name": "region", "required": False, "description": "UK region of hire"},
    {"name": "claimant_name", "required": False, "description": "Name of claimant"},
    {"name": "vehicle_registration", "required": False, "description": "Vehicle reg number"},
]


# =============================================================================
# Regex Pattern Definitions
# =============================================================================

# Currency pattern - matches UK monetary values
# Examples: £89.00, £1,234.56, 89.00, GBP 1234
CURRENCY_PATTERN = r'[£]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d{2})?)'

# Date patterns - various UK date formats
# Examples: 15/02/2026, 15-02-2026, 15 Feb 2026, February 15, 2026
DATE_PATTERNS = [
    r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # DD/MM/YYYY or DD-MM-YYYY
    r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}',  # DD Month YYYY
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}',  # Month DD, YYYY
]

# Vehicle registration pattern - UK format
# Examples: AB12 CDE, AB12CDE, A123 BCD
VEHICLE_REG_PATTERN = r'[A-Z]{2}\d{2}\s?[A-Z]{3}|[A-Z]\d{3}\s?[A-Z]{3}|[A-Z]{3}\s?\d{3}[A-Z]'


# =============================================================================
# Main Extraction Function
# =============================================================================

def extract_fields(
    document_text: str,
    tables: List[List[List[str]]] = None,
    key_value_pairs: Dict[str, str] = None
) -> Tuple[Dict[str, Any], float, List[str]]:
    """
    Extract structured claim fields from document content.
    
    This is the main entry point for field extraction. It:
    1. Applies regex patterns to find each field
    2. Uses table data to find rates and totals
    3. Uses key-value pairs for form-like fields
    4. Calculates extraction confidence
    5. Lists any missing required fields
    
    Args:
        document_text: Raw text extracted from the document.
        tables: Optional list of tables from Textract.
        key_value_pairs: Optional dict of form fields from Textract.
    
    Returns:
        Tuple containing:
        - extracted_fields (dict): All extracted field values
        - confidence_score (float): Extraction confidence (0.0-1.0)
        - missing_fields (list): Names of fields that couldn't be found
    
    Example:
        >>> text = "Invoice from ABC Hire\\nDaily rate: £89.00\\n..."
        >>> fields, confidence, missing = extract_fields(text)
        >>> print(f"Daily rate: £{fields['daily_rate']}")
        >>> print(f"Confidence: {confidence:.0%}")
    """
    # Initialize results
    extracted = {}
    
    # Normalize text for consistent matching
    normalized_text = _normalize_text(document_text)
    
    # Initialize with empty/default values
    key_value_pairs = key_value_pairs or {}
    tables = tables or []
    
    # Extract each field type
    # Order matters - some fields depend on others
    
    # 1. Extract dates first (needed to calculate hire_days)
    extracted["hire_start_date"] = _extract_hire_start_date(
        normalized_text, key_value_pairs
    )
    extracted["hire_end_date"] = _extract_hire_end_date(
        normalized_text, key_value_pairs
    )
    
    # 2. Calculate hire days from dates if not explicitly stated
    extracted["hire_days"] = _extract_hire_days(
        normalized_text,
        extracted["hire_start_date"],
        extracted["hire_end_date"]
    )
    
    # 3. Extract monetary values
    extracted["daily_rate"] = _extract_daily_rate(
        normalized_text, tables, key_value_pairs
    )
    extracted["total_claimed"] = _extract_total_claimed(
        normalized_text, tables, key_value_pairs
    )
    
    # 4. Extract vehicle information
    extracted["vehicle_class"] = _extract_vehicle_class(
        normalized_text, tables, key_value_pairs
    )
    extracted["vehicle_registration"] = _extract_vehicle_registration(normalized_text)
    
    # 5. Extract company and claimant
    extracted["hire_company"] = _extract_hire_company(
        normalized_text, key_value_pairs
    )
    extracted["claimant_name"] = _extract_claimant_name(
        normalized_text, key_value_pairs
    )
    
    # 6. Extract/infer region
    extracted["region"] = _extract_region(normalized_text)
    
    # Calculate confidence score based on what was found
    confidence_score = _calculate_confidence(extracted)
    
    # Identify missing required fields
    missing_fields = _get_missing_fields(extracted)
    
    # Log extraction results
    found_count = sum(1 for v in extracted.values() if v is not None)
    logger.info(f"Extracted {found_count}/{len(extracted)} fields (confidence: {confidence_score:.2f})")
    
    return extracted, confidence_score, missing_fields


# =============================================================================
# Date Extraction Functions
# =============================================================================

def _extract_hire_start_date(
    text: str,
    key_values: Dict[str, str]
) -> Optional[str]:
    """
    Extract the hire start date from document text.
    
    Looks for patterns like:
    - "Hire Start: 15/02/2026"
    - "Start Date: 15 February 2026"
    - "Hire commenced on 15/02/2026"
    
    Args:
        text: Normalized document text.
        key_values: Form field key-value pairs.
    
    Returns:
        str: Date in YYYY-MM-DD format, or None if not found.
    """
    # Try key-value pairs first (most reliable)
    start_keys = ["hire start", "start date", "from date", "commencement date"]
    for key, value in key_values.items():
        if any(sk in key.lower() for sk in start_keys):
            parsed = _parse_date(value)
            if parsed:
                return parsed
    
    # Try regex patterns on text
    # Look for start date indicators followed by a date
    start_patterns = [
        r'hire\s*(?:start|from|commenced?)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'start\s*date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'from\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'period\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s*(?:to|-)',
    ]
    
    for pattern in start_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            parsed = _parse_date(match.group(1))
            if parsed:
                return parsed
    
    return None


def _extract_hire_end_date(
    text: str,
    key_values: Dict[str, str]
) -> Optional[str]:
    """
    Extract the hire end date from document text.
    
    Looks for patterns like:
    - "Hire End: 01/03/2026"
    - "End Date: 1 March 2026"
    - "To: 01/03/2026"
    
    Args:
        text: Normalized document text.
        key_values: Form field key-value pairs.
    
    Returns:
        str: Date in YYYY-MM-DD format, or None if not found.
    """
    # Try key-value pairs first
    end_keys = ["hire end", "end date", "to date", "termination date"]
    for key, value in key_values.items():
        if any(ek in key.lower() for ek in end_keys):
            parsed = _parse_date(value)
            if parsed:
                return parsed
    
    # Try regex patterns
    end_patterns = [
        r'hire\s*(?:end|to|terminated?)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'end\s*date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'to\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'(?:to|-)\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
    ]
    
    for pattern in end_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            parsed = _parse_date(match.group(1))
            if parsed:
                return parsed
    
    return None


def _parse_date(date_string: str) -> Optional[str]:
    """
    Parse a date string into YYYY-MM-DD format.
    
    Uses dateutil for flexible parsing of various date formats.
    Assumes UK date format (DD/MM/YYYY) when ambiguous.
    
    Args:
        date_string: Date string to parse.
    
    Returns:
        str: Date in YYYY-MM-DD format, or None if parsing failed.
    """
    if not date_string:
        return None
    
    try:
        # Parse with dayfirst=True for UK format (DD/MM/YYYY)
        parsed = date_parser.parse(date_string, dayfirst=True)
        return parsed.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _extract_hire_days(
    text: str,
    start_date: Optional[str],
    end_date: Optional[str]
) -> Optional[int]:
    """
    Extract or calculate the number of hire days.
    
    First tries to find an explicit hire days value in the text.
    Falls back to calculating from start and end dates.
    
    Args:
        text: Normalized document text.
        start_date: Extracted start date (YYYY-MM-DD).
        end_date: Extracted end date (YYYY-MM-DD).
    
    Returns:
        int: Number of hire days, or None if cannot be determined.
    """
    # Try to find explicit hire days in text
    days_patterns = [
        r'(\d+)\s*days?\s*(?:hire|rental)',
        r'hire\s*(?:period|duration)\s*:?\s*(\d+)\s*days?',
        r'duration\s*:?\s*(\d+)\s*days?',
        r'number\s*of\s*days?\s*:?\s*(\d+)',
    ]
    
    for pattern in days_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                days = int(match.group(1))
                if 1 <= days <= 365:  # Sanity check
                    return days
            except ValueError:
                continue
    
    # Calculate from dates if available
    if start_date and end_date:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            days = (end - start).days + 1  # Inclusive of both days
            if 1 <= days <= 365:
                return days
        except ValueError:
            pass
    
    return None


# =============================================================================
# Monetary Value Extraction
# =============================================================================

def _extract_daily_rate(
    text: str,
    tables: List[List[List[str]]],
    key_values: Dict[str, str]
) -> Optional[float]:
    """
    Extract the daily hire rate from document content.
    
    This is a critical field - the comparison depends on it.
    We try multiple sources in order of reliability.
    
    Args:
        text: Normalized document text.
        tables: Tables extracted from document.
        key_values: Form field key-value pairs.
    
    Returns:
        float: Daily rate in GBP, or None if not found.
    """
    # Try key-value pairs first
    rate_keys = ["daily rate", "rate per day", "daily charge", "day rate"]
    for key, value in key_values.items():
        if any(rk in key.lower() for rk in rate_keys):
            rate = _parse_currency(value)
            if rate and 10 <= rate <= 500:  # Sanity check
                return rate
    
    # Try regex patterns on text
    rate_patterns = [
        r'daily\s*rate\s*:?\s*' + CURRENCY_PATTERN,
        r'rate\s*per\s*day\s*:?\s*' + CURRENCY_PATTERN,
        r'per\s*day\s*:?\s*' + CURRENCY_PATTERN,
        r'' + CURRENCY_PATTERN + r'\s*per\s*day',
        r'' + CURRENCY_PATTERN + r'\s*/\s*day',
    ]
    
    for pattern in rate_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            rate = _parse_currency(match.group(1))
            if rate and 10 <= rate <= 500:
                return rate
    
    # Try to find in tables (look for "daily rate" header)
    for table in tables:
        rate = _find_value_in_table(table, ["daily", "rate", "per day"])
        if rate:
            parsed = _parse_currency(rate)
            if parsed and 10 <= parsed <= 500:
                return parsed
    
    return None


def _extract_total_claimed(
    text: str,
    tables: List[List[List[str]]],
    key_values: Dict[str, str]
) -> Optional[float]:
    """
    Extract the total amount claimed from document content.
    
    Looks for the final total amount on the invoice.
    
    Args:
        text: Normalized document text.
        tables: Tables extracted from document.
        key_values: Form field key-value pairs.
    
    Returns:
        float: Total claimed in GBP, or None if not found.
    """
    # Try key-value pairs first
    total_keys = ["total", "amount due", "total due", "invoice total", "grand total"]
    for key, value in key_values.items():
        if any(tk in key.lower() for tk in total_keys):
            total = _parse_currency(value)
            if total and total > 0:
                return total
    
    # Try regex patterns
    total_patterns = [
        r'total\s*(?:claimed|due|amount)?\s*:?\s*' + CURRENCY_PATTERN,
        r'grand\s*total\s*:?\s*' + CURRENCY_PATTERN,
        r'amount\s*due\s*:?\s*' + CURRENCY_PATTERN,
        r'invoice\s*total\s*:?\s*' + CURRENCY_PATTERN,
        r'total\s*hire\s*(?:charges?)?\s*:?\s*' + CURRENCY_PATTERN,
    ]
    
    for pattern in total_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            total = _parse_currency(match.group(1))
            if total and total > 0:
                return total
    
    # Try to find in tables
    for table in tables:
        total = _find_value_in_table(table, ["total", "amount due"])
        if total:
            parsed = _parse_currency(total)
            if parsed and parsed > 0:
                return parsed
    
    return None


def _parse_currency(value: str) -> Optional[float]:
    """
    Parse a currency string to a float value.
    
    Handles various formats:
    - £89.00
    - 89.00
    - £1,234.56
    - GBP 1234.56
    
    Args:
        value: Currency string to parse.
    
    Returns:
        float: Numeric value, or None if parsing failed.
    """
    if not value:
        return None
    
    try:
        # Remove currency symbols and whitespace
        cleaned = value.replace("£", "").replace("GBP", "").replace(",", "").strip()
        return float(cleaned)
    except (ValueError, TypeError):
        return None


# =============================================================================
# Vehicle Information Extraction
# =============================================================================

def _extract_vehicle_class(
    text: str,
    tables: List[List[List[str]]],
    key_values: Dict[str, str]
) -> Optional[str]:
    """
    Extract the vehicle class/category from document content.
    
    Vehicle classes follow industry standards:
    - GROUP_A through GROUP_J (size categories)
    - SUV, MPV, VAN for special types
    
    Args:
        text: Normalized document text.
        tables: Tables extracted from document.
        key_values: Form field key-value pairs.
    
    Returns:
        str: Vehicle class constant, or None if not found.
    """
    # Try key-value pairs first
    class_keys = ["vehicle class", "vehicle group", "group", "category", "vehicle type"]
    for key, value in key_values.items():
        if any(ck in key.lower() for ck in class_keys):
            vehicle_class = _normalize_vehicle_class(value)
            if vehicle_class:
                return vehicle_class
    
    # Try to find explicit group references
    group_pattern = r'group\s*([a-j])\b'
    match = re.search(group_pattern, text, re.IGNORECASE)
    if match:
        return f"GROUP_{match.group(1).upper()}"
    
    # Look for vehicle type keywords
    type_mappings = [
        (["suv", "sports utility", "4x4", "crossover"], VehicleClass.SUV),
        (["mpv", "people carrier", "minivan", "multi-purpose"], VehicleClass.MPV),
        (["van", "transit", "commercial"], VehicleClass.VAN),
        (["mini", "city car", "small car"], VehicleClass.GROUP_A),
        (["economy", "supermini"], VehicleClass.GROUP_B),
        (["compact", "small family"], VehicleClass.GROUP_C),
        (["medium", "family car"], VehicleClass.GROUP_D),
        (["large", "executive"], VehicleClass.GROUP_E),
        (["luxury", "premium"], VehicleClass.GROUP_H),
        (["prestige"], VehicleClass.GROUP_J),
    ]
    
    text_lower = text.lower()
    for keywords, vehicle_class in type_mappings:
        if any(kw in text_lower for kw in keywords):
            return vehicle_class
    
    return None


def _normalize_vehicle_class(value: str) -> Optional[str]:
    """
    Normalize a vehicle class value to our standard format.
    
    Args:
        value: Raw vehicle class string.
    
    Returns:
        str: Normalized vehicle class constant.
    """
    if not value:
        return None
    
    value_upper = value.upper().strip()
    
    # Check for direct group match
    if re.match(r'GROUP[_\s]?[A-J]', value_upper):
        letter = re.search(r'[A-J]', value_upper).group()
        return f"GROUP_{letter}"
    
    # Check for single letter
    if re.match(r'^[A-J]$', value_upper):
        return f"GROUP_{value_upper}"
    
    # Check for special types
    special_types = {
        "SUV": VehicleClass.SUV,
        "MPV": VehicleClass.MPV,
        "VAN": VehicleClass.VAN,
    }
    for key, val in special_types.items():
        if key in value_upper:
            return val
    
    return None


def _extract_vehicle_registration(text: str) -> Optional[str]:
    """
    Extract UK vehicle registration number from text.
    
    UK registrations follow patterns like:
    - AB12 CDE (new format)
    - A123 BCD (prefix format)
    - ABC 123D (suffix format)
    
    Args:
        text: Normalized document text.
    
    Returns:
        str: Vehicle registration, or None if not found.
    """
    match = re.search(VEHICLE_REG_PATTERN, text.upper())
    if match:
        return match.group().replace(" ", "")
    return None


# =============================================================================
# Company and Claimant Extraction
# =============================================================================

def _extract_hire_company(
    text: str,
    key_values: Dict[str, str]
) -> Optional[str]:
    """
    Extract the credit hire company name from document.
    
    Common patterns:
    - Company letterhead at top of document
    - "Invoice from: ABC Hire Ltd"
    - Form field labeled "Supplier" or "Company"
    
    Args:
        text: Normalized document text.
        key_values: Form field key-value pairs.
    
    Returns:
        str: Company name, or None if not found.
    """
    # Try key-value pairs
    company_keys = ["company", "supplier", "hire company", "from", "credit hire"]
    for key, value in key_values.items():
        if any(ck in key.lower() for ck in company_keys):
            if value and len(value) > 2:
                return value.strip()
    
    # Try regex patterns
    company_patterns = [
        r'(?:from|supplier|company)\s*:?\s*([A-Z][A-Za-z\s&]+(?:Ltd|Limited|PLC|LLP)?)',
        r'^([A-Z][A-Za-z\s&]+(?:Ltd|Limited|PLC|LLP))',  # First line of document
        r'([A-Z][A-Za-z\s&]+Hire\s*(?:Ltd|Limited)?)',
    ]
    
    for pattern in company_patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            company = match.group(1).strip()
            if len(company) > 2 and len(company) < 100:
                return company
    
    return None


def _extract_claimant_name(
    text: str,
    key_values: Dict[str, str]
) -> Optional[str]:
    """
    Extract the claimant's name from document.
    
    Args:
        text: Normalized document text.
        key_values: Form field key-value pairs.
    
    Returns:
        str: Claimant name, or None if not found.
    """
    # Try key-value pairs
    claimant_keys = ["claimant", "customer", "client", "name", "hirer"]
    for key, value in key_values.items():
        if any(ck in key.lower() for ck in claimant_keys):
            if value and len(value) > 2:
                return value.strip()
    
    # Try regex patterns
    claimant_patterns = [
        r'claimant\s*:?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)',
        r'customer\s*:?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)',
        r'hirer\s*:?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)',
    ]
    
    for pattern in claimant_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    
    return None


# =============================================================================
# Region Extraction
# =============================================================================

def _extract_region(text: str) -> Optional[str]:
    """
    Extract or infer the UK region from document.
    
    Looks for explicit region mentions or postcodes to determine
    the geographic area for rate matching.
    
    Args:
        text: Normalized document text.
    
    Returns:
        str: Region constant, or None if cannot be determined.
    """
    text_lower = text.lower()
    
    # Region keywords mapping
    region_mappings = [
        (["london", "greater london"], Region.LONDON),
        (["manchester", "liverpool", "preston", "bolton"], Region.NORTH_WEST),
        (["birmingham", "coventry", "wolverhampton"], Region.WEST_MIDLANDS),
        (["leeds", "sheffield", "bradford", "hull"], Region.YORKSHIRE),
        (["newcastle", "sunderland", "durham"], Region.NORTH_EAST),
        (["nottingham", "leicester", "derby"], Region.EAST_MIDLANDS),
        (["bristol", "bath", "exeter", "plymouth"], Region.SOUTH_WEST),
        (["brighton", "southampton", "portsmouth"], Region.SOUTH_EAST),
        (["cambridge", "norwich", "ipswich"], Region.EAST_ANGLIA),
        (["cardiff", "swansea", "newport"], Region.WALES),
        (["edinburgh", "glasgow", "aberdeen"], Region.SCOTLAND),
        (["belfast"], Region.NORTHERN_IRELAND),
    ]
    
    for keywords, region in region_mappings:
        if any(kw in text_lower for kw in keywords):
            return region
    
    # Try to infer from postcode
    postcode_region = _infer_region_from_postcode(text)
    if postcode_region:
        return postcode_region
    
    return None


def _infer_region_from_postcode(text: str) -> Optional[str]:
    """
    Infer UK region from postcode prefix.
    
    UK postcodes start with area letters that indicate region.
    
    Args:
        text: Document text to search.
    
    Returns:
        str: Region constant, or None if no postcode found.
    """
    # UK postcode pattern (simplified)
    postcode_match = re.search(r'\b([A-Z]{1,2})\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b', text.upper())
    
    if not postcode_match:
        return None
    
    prefix = postcode_match.group(1)
    
    # Postcode prefix to region mapping
    # This is simplified - real mapping is more complex
    postcode_regions = {
        "E": Region.LONDON, "EC": Region.LONDON, "N": Region.LONDON,
        "NW": Region.LONDON, "SE": Region.LONDON, "SW": Region.LONDON,
        "W": Region.LONDON, "WC": Region.LONDON,
        "M": Region.NORTH_WEST, "L": Region.NORTH_WEST, "PR": Region.NORTH_WEST,
        "BL": Region.NORTH_WEST, "WN": Region.NORTH_WEST,
        "B": Region.WEST_MIDLANDS, "CV": Region.WEST_MIDLANDS, "WV": Region.WEST_MIDLANDS,
        "LS": Region.YORKSHIRE, "S": Region.YORKSHIRE, "BD": Region.YORKSHIRE,
        "HU": Region.YORKSHIRE,
        "NE": Region.NORTH_EAST, "SR": Region.NORTH_EAST, "DH": Region.NORTH_EAST,
        "NG": Region.EAST_MIDLANDS, "LE": Region.EAST_MIDLANDS, "DE": Region.EAST_MIDLANDS,
        "BS": Region.SOUTH_WEST, "BA": Region.SOUTH_WEST, "EX": Region.SOUTH_WEST,
        "BN": Region.SOUTH_EAST, "SO": Region.SOUTH_EAST, "PO": Region.SOUTH_EAST,
        "CB": Region.EAST_ANGLIA, "NR": Region.EAST_ANGLIA, "IP": Region.EAST_ANGLIA,
        "CF": Region.WALES, "SA": Region.WALES, "NP": Region.WALES,
        "EH": Region.SCOTLAND, "G": Region.SCOTLAND, "AB": Region.SCOTLAND,
        "BT": Region.NORTHERN_IRELAND,
    }
    
    return postcode_regions.get(prefix)


# =============================================================================
# Helper Functions
# =============================================================================

def _normalize_text(text: str) -> str:
    """
    Normalize document text for consistent matching.
    
    - Collapses multiple whitespace
    - Removes control characters
    - Standardizes line endings
    
    Args:
        text: Raw document text.
    
    Returns:
        str: Normalized text.
    """
    if not text:
        return ""
    
    # Replace various whitespace with single space
    text = re.sub(r'[\t\r]+', ' ', text)
    
    # Collapse multiple spaces
    text = re.sub(r' +', ' ', text)
    
    # Normalize line endings
    text = re.sub(r'\n+', '\n', text)
    
    return text.strip()


def _find_value_in_table(
    table: List[List[str]],
    keywords: List[str]
) -> Optional[str]:
    """
    Find a value in a table that's associated with given keywords.
    
    Searches table cells for keywords and returns the adjacent cell value.
    
    Args:
        table: 2D list of table cells.
        keywords: Keywords to search for.
    
    Returns:
        str: The adjacent cell value, or None if not found.
    """
    for row in table:
        for i, cell in enumerate(row):
            cell_lower = cell.lower()
            if any(kw in cell_lower for kw in keywords):
                # Return next cell in row if exists
                if i + 1 < len(row):
                    return row[i + 1]
    return None


def _calculate_confidence(extracted: Dict[str, Any]) -> float:
    """
    Calculate extraction confidence score.
    
    The confidence score indicates how complete and reliable
    the extraction is. Based on:
    - Which fields were successfully extracted
    - Weighted by field importance
    
    Args:
        extracted: Dictionary of extracted fields.
    
    Returns:
        float: Confidence score between 0.0 and 1.0.
    """
    confidence = 0.0
    
    # Use configured weights
    if extracted.get("hire_start_date") and extracted.get("hire_end_date"):
        confidence += settings.CONFIDENCE_WEIGHT_HIRE_DATES
    elif extracted.get("hire_days"):
        confidence += settings.CONFIDENCE_WEIGHT_HIRE_DATES * 0.5
    
    if extracted.get("vehicle_class"):
        confidence += settings.CONFIDENCE_WEIGHT_VEHICLE_CLASS
    
    if extracted.get("daily_rate"):
        confidence += settings.CONFIDENCE_WEIGHT_DAILY_RATE
    
    if extracted.get("total_claimed"):
        confidence += settings.CONFIDENCE_WEIGHT_TOTAL_CLAIMED
    
    if extracted.get("hire_company"):
        confidence += settings.CONFIDENCE_WEIGHT_HIRE_COMPANY
    
    return min(confidence, 1.0)  # Cap at 1.0


def _get_missing_fields(extracted: Dict[str, Any]) -> List[str]:
    """
    Identify which required fields are missing.
    
    Args:
        extracted: Dictionary of extracted fields.
    
    Returns:
        list: Names of missing required fields.
    """
    missing = []
    
    for field in REQUIRED_FIELDS:
        if field["required"]:
            value = extracted.get(field["name"])
            if value is None:
                missing.append(field["name"])
    
    return missing


# =============================================================================
# Validation Functions
# =============================================================================

def validate_extraction(
    extracted: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Validate extracted data for consistency and reasonableness.
    
    Checks:
    - Daily rate is within realistic range
    - Total matches daily rate * days (approximately)
    - Dates are in the past (not future claims)
    - Hire period is reasonable (not > 1 year)
    
    Args:
        extracted: Dictionary of extracted fields.
    
    Returns:
        Tuple containing:
        - valid (bool): True if data passes validation
        - warnings (list): List of validation warning messages
    """
    warnings = []
    
    # Check daily rate range
    daily_rate = extracted.get("daily_rate")
    if daily_rate:
        if daily_rate < 10:
            warnings.append(f"Daily rate (£{daily_rate}) seems too low")
        elif daily_rate > 500:
            warnings.append(f"Daily rate (£{daily_rate}) seems unusually high")
    
    # Check total vs calculated
    if daily_rate and extracted.get("hire_days"):
        expected_total = daily_rate * extracted["hire_days"]
        actual_total = extracted.get("total_claimed", 0)
        
        if actual_total:
            variance = abs(actual_total - expected_total) / expected_total
            if variance > 0.2:  # More than 20% difference
                warnings.append(
                    f"Total (£{actual_total}) differs significantly from "
                    f"calculated (£{expected_total:.2f})"
                )
    
    # Check hire period length
    hire_days = extracted.get("hire_days")
    if hire_days:
        if hire_days > 365:
            warnings.append(f"Hire period ({hire_days} days) exceeds 1 year")
        elif hire_days > 90:
            warnings.append(f"Hire period ({hire_days} days) is unusually long")
    
    # Check dates are not in future
    end_date = extracted.get("hire_end_date")
    if end_date:
        try:
            end = datetime.strptime(end_date, "%Y-%m-%d")
            if end > datetime.now():
                warnings.append("Hire end date is in the future")
        except ValueError:
            pass
    
    is_valid = len(warnings) == 0
    
    return is_valid, warnings
