"""
extractor.py - LLM-Based Field Extraction from Documents

WHY THIS FILE EXISTS:
- Extract structured data from unstructured court pack documents
- Use LLM instead of fragile regex patterns
- LLM understands context ("daily rate" vs "total cost")
- Returns clean, validated data for downstream processing

HOW IT WORKS:
1. Take raw document text (from document_parser.py)
2. Send to LLM with extraction prompt
3. LLM returns JSON with extracted fields
4. Validate and clean the extracted data

WHAT WE EXTRACT:
- Hire dates (start, end, duration)
- Vehicle details (make, model, group)
- Rates claimed (daily, weekly, total)
- Company information
- Claim reference numbers
- Any additional charges

WHY LLM INSTEAD OF REGEX:
- "£55 per day" vs "daily rate of 55 pounds" vs "55.00/day"
- LLM handles all variations naturally
- Regex would need dozens of patterns and still miss edge cases
"""

import logging
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, date
from dateutil import parser as date_parser

# Import from sibling modules using package-relative imports
from src.llm_client import OllamaClient, get_llm_client
from src.config import settings

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class VehicleInfo:
    """
    Extracted vehicle information.
    
    Attributes:
        make: Manufacturer (e.g., BMW, Ford)
        model: Model name (e.g., 3 Series, Focus)
        group: Vehicle group classification (e.g., A, B, C, D)
        registration: Registration plate if found
        year: Year of manufacture if found
    """
    make: Optional[str] = None
    model: Optional[str] = None
    group: Optional[str] = None
    registration: Optional[str] = None
    year: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class HirePeriod:
    """
    Extracted hire period information.
    
    Attributes:
        start_date: When hire began
        end_date: When hire ended
        duration_days: Number of days hired
    """
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    duration_days: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with ISO date strings."""
        result = {}
        if self.start_date:
            result["start_date"] = self.start_date.isoformat()
        if self.end_date:
            result["end_date"] = self.end_date.isoformat()
        if self.duration_days:
            result["duration_days"] = self.duration_days
        return result


@dataclass
class RateInfo:
    """
    Extracted rate/cost information.
    
    Attributes:
        daily_rate: Rate per day in GBP
        weekly_rate: Rate per week if specified
        total_cost: Total claim amount
        additional_charges: Extra charges (CDW, delivery, etc.)
        currency: Currency code (default GBP)
    """
    daily_rate: Optional[float] = None
    weekly_rate: Optional[float] = None
    total_cost: Optional[float] = None
    additional_charges: Dict[str, float] = field(default_factory=dict)
    currency: str = "GBP"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"currency": self.currency}
        if self.daily_rate is not None:
            result["daily_rate"] = self.daily_rate
        if self.weekly_rate is not None:
            result["weekly_rate"] = self.weekly_rate
        if self.total_cost is not None:
            result["total_cost"] = self.total_cost
        if self.additional_charges:
            result["additional_charges"] = self.additional_charges
        return result


@dataclass
class ExtractedClaim:
    """
    All extracted information from a court pack document.
    
    This is the MAIN OUTPUT of the extractor.
    """
    # Claim identification
    claim_reference: Optional[str] = None
    claimant_name: Optional[str] = None
    defendant_name: Optional[str] = None
    
    # Hire company
    hire_company: Optional[str] = None
    
    # Vehicle details
    vehicle: VehicleInfo = field(default_factory=VehicleInfo)
    
    # Hire period
    hire_period: HirePeriod = field(default_factory=HirePeriod)
    
    # Rates and costs
    rates: RateInfo = field(default_factory=RateInfo)
    
    # Additional extracted data
    accident_date: Optional[date] = None
    accident_location: Optional[str] = None
    
    # Metadata
    extraction_confidence: float = 0.0
    raw_extractions: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "claim_reference": self.claim_reference,
            "claimant_name": self.claimant_name,
            "defendant_name": self.defendant_name,
            "hire_company": self.hire_company,
            "vehicle": self.vehicle.to_dict(),
            "hire_period": self.hire_period.to_dict(),
            "rates": self.rates.to_dict(),
            "accident_date": self.accident_date.isoformat() if self.accident_date else None,
            "accident_location": self.accident_location,
            "extraction_confidence": self.extraction_confidence
        }


class ClaimExtractor:
    """
    Extracts structured claim information from document text using LLM.
    
    USAGE:
        extractor = ClaimExtractor()
        
        # From document text
        claim = extractor.extract(document_text)
        
        print(f"Daily rate: £{claim.rates.daily_rate}")
        print(f"Vehicle: {claim.vehicle.make} {claim.vehicle.model}")
        print(f"Hire period: {claim.hire_period.duration_days} days")
    
    HOW IT WORKS:
    1. Send document text to LLM with extraction prompt
    2. LLM returns JSON with extracted fields
    3. Parse and validate the JSON
    4. Convert to ExtractedClaim dataclass
    """
    
    def __init__(self, llm_client: Optional[OllamaClient] = None):
        """
        Initialize the extractor.
        
        Args:
            llm_client: LLM client to use (default: Ollama)
        """
        self.llm_client = llm_client or get_llm_client()
        
        logger.info("ClaimExtractor initialized")
    
    def extract(self, document_text: str) -> ExtractedClaim:
        """
        Extract claim information from document text.
        
        This is the MAIN METHOD.
        
        Args:
            document_text: Raw text extracted from PDF
            
        Returns:
            ExtractedClaim with all extracted fields
            
        EXAMPLE:
            text = '''
            CREDIT HIRE INVOICE
            Claim Ref: CH-2024-12345
            Vehicle: BMW 320d (Group D)
            Hire Period: 15/01/2024 to 29/01/2024 (14 days)
            Daily Rate: £65.00
            Total: £910.00
            '''
            
            claim = extractor.extract(text)
            print(claim.rates.daily_rate)  # 65.0
        """
        logger.info("Starting claim extraction...")
        
        # Step 1: Use LLM to extract raw JSON
        raw_extraction = self._extract_with_llm(document_text)
        
        # Step 2: Parse and validate into dataclasses
        claim = self._parse_extraction(raw_extraction)
        
        # Step 3: Post-process and calculate derived values
        claim = self._post_process(claim)
        
        # Store raw extractions for debugging
        claim.raw_extractions = raw_extraction
        
        return claim
    
    def _extract_with_llm(self, document_text: str) -> Dict[str, Any]:
        """
        Send document to LLM for extraction.
        
        Returns raw JSON dict from LLM.
        """
        # Build the extraction prompt
        system_prompt = self._get_extraction_system_prompt()
        user_prompt = self._get_extraction_user_prompt(document_text)
        
        # Call LLM with JSON mode
        result = self.llm_client.generate_json(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0  # Zero temperature for consistent extraction
        )
        
        if not result:
            logger.warning("LLM returned empty extraction")
            return {}
        
        return result
    
    def _get_extraction_system_prompt(self) -> str:
        """
        System prompt that instructs LLM on extraction.
        
        KEY POINTS:
        - Be specific about field names
        - Specify formats (dates, numbers)
        - Tell it to use null for missing values
        """
        return """You are a document data extraction specialist for motor insurance claims.

Your task is to extract specific fields from court pack documents and return them as JSON.

EXTRACTION RULES:
1. Extract ONLY what is explicitly stated in the document
2. Use null for any field that is not clearly present
3. For dates, use ISO format: YYYY-MM-DD
4. For currency amounts, extract as numbers without symbols (65.00, not £65.00)
5. For vehicle groups, normalize to single letters: A, B, C, D, E, F, G, H, I
6. Be precise - don't guess or infer values

OUTPUT FORMAT - Return this exact JSON structure:
{
    "claim_reference": "string or null",
    "claimant_name": "string or null",
    "defendant_name": "string or null",
    "hire_company": "string or null",
    "vehicle": {
        "make": "string or null",
        "model": "string or null",
        "group": "single letter A-I or null",
        "registration": "string or null",
        "year": "number or null"
    },
    "hire_period": {
        "start_date": "YYYY-MM-DD or null",
        "end_date": "YYYY-MM-DD or null",
        "duration_days": "number or null"
    },
    "rates": {
        "daily_rate": "number or null",
        "weekly_rate": "number or null",
        "total_cost": "number or null",
        "additional_charges": {"charge_name": "amount"} or {}
    },
    "accident_date": "YYYY-MM-DD or null",
    "accident_location": "string or null"
}"""
    
    def _get_extraction_user_prompt(self, document_text: str) -> str:
        """
        Build the user prompt with document text.
        """
        # Truncate very long documents to avoid context limits
        max_chars = 8000
        if len(document_text) > max_chars:
            document_text = document_text[:max_chars] + "\n\n[Document truncated...]"
        
        return f"""Extract the claim information from this document:

--- DOCUMENT START ---
{document_text}
--- DOCUMENT END ---

Return the extracted data as JSON."""
    
    def _parse_extraction(self, raw: Dict[str, Any]) -> ExtractedClaim:
        """
        Parse raw LLM output into ExtractedClaim dataclass.
        
        Handles missing fields and type conversion.
        """
        claim = ExtractedClaim()
        
        # Simple string fields
        claim.claim_reference = raw.get("claim_reference")
        claim.claimant_name = raw.get("claimant_name")
        claim.defendant_name = raw.get("defendant_name")
        claim.hire_company = raw.get("hire_company")
        claim.accident_location = raw.get("accident_location")
        
        # Parse vehicle info
        vehicle_data = raw.get("vehicle", {})
        if vehicle_data:
            claim.vehicle = VehicleInfo(
                make=vehicle_data.get("make"),
                model=vehicle_data.get("model"),
                group=self._normalize_vehicle_group(vehicle_data.get("group")),
                registration=vehicle_data.get("registration"),
                year=self._parse_int(vehicle_data.get("year"))
            )
        
        # Parse hire period
        period_data = raw.get("hire_period", {})
        if period_data:
            claim.hire_period = HirePeriod(
                start_date=self._parse_date(period_data.get("start_date")),
                end_date=self._parse_date(period_data.get("end_date")),
                duration_days=self._parse_int(period_data.get("duration_days"))
            )
        
        # Parse rates
        rates_data = raw.get("rates", {})
        if rates_data:
            claim.rates = RateInfo(
                daily_rate=self._parse_float(rates_data.get("daily_rate")),
                weekly_rate=self._parse_float(rates_data.get("weekly_rate")),
                total_cost=self._parse_float(rates_data.get("total_cost")),
                additional_charges=rates_data.get("additional_charges", {})
            )
        
        # Parse accident date
        claim.accident_date = self._parse_date(raw.get("accident_date"))
        
        return claim
    
    def _post_process(self, claim: ExtractedClaim) -> ExtractedClaim:
        """
        Post-process the claim to calculate derived values.
        
        E.g., calculate duration from dates if not provided.
        """
        # Calculate duration if we have dates but no duration
        if (claim.hire_period.start_date and 
            claim.hire_period.end_date and 
            not claim.hire_period.duration_days):
            
            delta = claim.hire_period.end_date - claim.hire_period.start_date
            claim.hire_period.duration_days = delta.days
        
        # Calculate daily rate from total if not provided
        if (claim.rates.total_cost and 
            claim.hire_period.duration_days and 
            not claim.rates.daily_rate):
            
            claim.rates.daily_rate = round(
                claim.rates.total_cost / claim.hire_period.duration_days, 
                2
            )
        
        # Calculate total from daily rate if not provided
        if (claim.rates.daily_rate and 
            claim.hire_period.duration_days and 
            not claim.rates.total_cost):
            
            claim.rates.total_cost = round(
                claim.rates.daily_rate * claim.hire_period.duration_days, 
                2
            )
        
        # Calculate extraction confidence based on completeness
        claim.extraction_confidence = self._calculate_confidence(claim)
        
        return claim
    
    def _calculate_confidence(self, claim: ExtractedClaim) -> float:
        """
        Calculate confidence score based on extraction completeness.
        
        Returns 0.0 to 1.0 based on how many key fields were extracted.
        """
        # Define key fields and their weights
        checks = [
            (claim.vehicle.group is not None, 0.2),  # Vehicle group is critical
            (claim.rates.daily_rate is not None, 0.25),  # Daily rate is critical
            (claim.hire_period.duration_days is not None, 0.15),
            (claim.hire_period.start_date is not None, 0.1),
            (claim.hire_company is not None, 0.1),
            (claim.vehicle.make is not None, 0.1),
            (claim.claim_reference is not None, 0.1),
        ]
        
        total_weight = sum(weight for _, weight in checks)
        achieved_weight = sum(weight for passed, weight in checks if passed)
        
        return round(achieved_weight / total_weight, 2)
    
    # -------------------------------------------------------------------------
    # Helper methods for parsing/cleaning
    # -------------------------------------------------------------------------
    
    def _parse_date(self, value: Any) -> Optional[date]:
        """
        Parse a date value flexibly.
        
        Handles: "2024-01-15", "15/01/2024", "January 15, 2024", etc.
        """
        if value is None:
            return None
        
        if isinstance(value, date):
            return value
        
        if isinstance(value, datetime):
            return value.date()
        
        if isinstance(value, str):
            try:
                # dateutil.parser handles many formats
                parsed = date_parser.parse(value, dayfirst=True)  # UK format
                return parsed.date()
            except Exception:
                logger.debug(f"Could not parse date: {value}")
                return None
        
        return None
    
    def _parse_float(self, value: Any) -> Optional[float]:
        """
        Parse a float value, handling strings with currency symbols.
        """
        if value is None:
            return None
        
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Remove currency symbols and commas
            cleaned = re.sub(r'[£$€,]', '', value.strip())
            try:
                return float(cleaned)
            except ValueError:
                return None
        
        return None
    
    def _parse_int(self, value: Any) -> Optional[int]:
        """Parse an integer value."""
        if value is None:
            return None
        
        if isinstance(value, int):
            return value
        
        if isinstance(value, float):
            return int(value)
        
        if isinstance(value, str):
            try:
                return int(float(value))
            except ValueError:
                return None
        
        return None
    
    def _normalize_vehicle_group(self, group: Any) -> Optional[str]:
        """
        Normalize vehicle group to single uppercase letter.
        
        Handles: "Group C", "C", "c", "GROUP_C", etc.
        """
        if group is None:
            return None
        
        group_str = str(group).upper().strip()
        
        # Extract single letter from various formats
        # "GROUP C" -> "C", "GROUP_D" -> "D", "C" -> "C"
        match = re.search(r'[A-I]', group_str)
        if match:
            return match.group(0)
        
        return None


# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------
_default_extractor: Optional[ClaimExtractor] = None


def get_extractor() -> ClaimExtractor:
    """Get the default extractor (singleton)."""
    global _default_extractor
    
    if _default_extractor is None:
        _default_extractor = ClaimExtractor()
    
    return _default_extractor


def extract_claim(document_text: str) -> ExtractedClaim:
    """
    Quick function to extract claim from text.
    
    USAGE:
        from extractor import extract_claim
        
        claim = extract_claim(pdf_text)
        print(f"Rate: £{claim.rates.daily_rate}/day")
    """
    extractor = get_extractor()
    return extractor.extract(document_text)