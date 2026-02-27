# =============================================================================
# ClaimsNOW - AWS Bedrock Integration Module
# =============================================================================
# This module handles AI-powered document analysis using Claude via Bedrock.
#
# Bedrock provides access to Claude 3 models for:
# 1. Intelligent field extraction when regex/patterns fail
# 2. Understanding complex or poorly formatted documents
# 3. Natural language interpretation of invoice content
#
# This is the "agentic" layer of ClaimsNOW - when deterministic
# extraction fails, we escalate to AI for intelligent understanding.
# =============================================================================

import json
import logging
from typing import Dict, Optional, Tuple, Any
import boto3
from botocore.exceptions import ClientError

from src.config import settings

# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# Bedrock Client Initialization
# =============================================================================

def get_bedrock_client():
    """
    Create and return a Bedrock Runtime client.
    
    We use bedrock-runtime (not bedrock) because we're invoking
    models, not managing them.
    
    Returns:
        boto3.client: Configured Bedrock Runtime client.
    """
    client_config = {
        "service_name": "bedrock-runtime",
        "region_name": settings.AWS_REGION,
    }
    
    if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
        client_config["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
        client_config["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
    
    return boto3.client(**client_config)


# =============================================================================
# Field Extraction Prompt
# =============================================================================

# This prompt is carefully crafted to extract insurance claim data
# from court pack documents. It's structured to:
# 1. Clearly define what fields we need
# 2. Specify the exact output format (JSON)
# 3. Handle missing data gracefully
# 4. Avoid hallucination by asking for null when not found

EXTRACTION_PROMPT_TEMPLATE = """You are an expert at extracting structured data from UK motor insurance hire invoices and court pack documents.

Your task is to extract specific fields from the document text below. Return ONLY a valid JSON object with the extracted data.

## Fields to Extract

1. **hire_start_date**: The date when the vehicle hire period began (format: YYYY-MM-DD)
2. **hire_end_date**: The date when the vehicle hire period ended (format: YYYY-MM-DD)
3. **hire_days**: The total number of days the vehicle was hired (integer)
4. **vehicle_class**: The vehicle category/group. Common values:
   - GROUP_A through GROUP_J (standard hire groups)
   - SUV, MPV, VAN for special categories
   - If described as "small", "economy" → GROUP_B
   - If described as "medium", "family" → GROUP_C or GROUP_D
   - If described as "large", "executive" → GROUP_E or GROUP_F
   - If described as "prestige", "luxury" → GROUP_H or higher
5. **daily_rate**: The daily hire rate in GBP (number, e.g., 89.00)
6. **total_claimed**: The total amount claimed in GBP (number, e.g., 1246.00)
7. **hire_company**: The name of the credit hire company
8. **claimant_name**: The name of the person making the claim (if mentioned)
9. **vehicle_registration**: The registration number of the hired vehicle (if mentioned)
10. **region**: The UK region where the hire took place. Use one of:
    - LONDON, SOUTH_EAST, SOUTH_WEST, EAST_ANGLIA
    - EAST_MIDLANDS, WEST_MIDLANDS, NORTH_WEST, NORTH_EAST
    - YORKSHIRE, WALES, SCOTLAND, NORTHERN_IRELAND
    - UNKNOWN if region cannot be determined

## Important Instructions

- Return ONLY valid JSON, no other text or explanation
- Use null for any field that cannot be found in the document
- Convert all monetary values to numbers (remove £ symbols and commas)
- Ensure dates are in YYYY-MM-DD format
- If multiple values exist for a field, use the most prominent/relevant one
- Be precise - do not guess or hallucinate values that aren't in the document

## Document Text

{document_text}

## Your Response (JSON only)
"""


# =============================================================================
# Main Extraction Function
# =============================================================================

def extract_fields_with_claude(
    document_text: str,
    partial_data: Dict[str, Any] = None
) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """
    Extract claim fields from document text using Claude AI.
    
    This function sends document text to Claude and asks it to
    identify and extract specific fields related to insurance claims.
    
    Use this when:
    - Regex-based extraction failed to find all required fields
    - Document format is unusual or complex
    - Higher accuracy is needed for ambiguous cases
    
    Args:
        document_text: The raw text extracted from the document (via Textract).
        partial_data: Optional dict of fields already extracted by regex.
            Claude will focus on filling in missing fields.
    
    Returns:
        Tuple containing:
        - success (bool): True if extraction succeeded
        - fields (dict|None): Extracted fields if successful:
            - hire_start_date: str (YYYY-MM-DD)
            - hire_end_date: str (YYYY-MM-DD)
            - hire_days: int
            - vehicle_class: str
            - daily_rate: float
            - total_claimed: float
            - hire_company: str
            - claimant_name: str (or None)
            - vehicle_registration: str (or None)
            - region: str
        - error (str|None): Error message if failed
    
    Example:
        >>> text = "Invoice from ABC Hire Ltd\\nDaily rate: £89.00\\n..."
        >>> success, fields, error = extract_fields_with_claude(text)
        >>> if success:
        ...     print(f"Daily rate: £{fields['daily_rate']}")
    """
    # Check if document text is provided
    if not document_text or not document_text.strip():
        return False, None, "Empty document text provided"
    
    # Truncate very long documents to stay within token limits
    # Claude 3 Sonnet has ~200k context, but we'll use a reasonable limit
    max_chars = 50000
    if len(document_text) > max_chars:
        logger.warning(f"Document text truncated from {len(document_text)} to {max_chars} chars")
        document_text = document_text[:max_chars]
    
    # Build the prompt
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(document_text=document_text)
    
    # If we have partial data, add context about what's already known
    if partial_data:
        known_fields = {k: v for k, v in partial_data.items() if v is not None}
        if known_fields:
            prompt += f"\n\nNote: Some fields have already been extracted. Verify and complete:\n{json.dumps(known_fields, indent=2)}"
    
    try:
        # Get Bedrock client
        bedrock = get_bedrock_client()
        
        # Prepare the request body for Claude
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": settings.BEDROCK_MAX_TOKENS,
            "temperature": settings.BEDROCK_TEMPERATURE,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        # Invoke Claude via Bedrock
        response = bedrock.invoke_model(
            modelId=settings.BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body)
        )
        
        # Parse the response
        response_body = json.loads(response["body"].read())
        
        # Extract the text content from Claude's response
        content = response_body.get("content", [])
        if not content:
            return False, None, "Empty response from Claude"
        
        # Get the text from the first content block
        response_text = content[0].get("text", "")
        
        # Parse the JSON from Claude's response
        # Claude should return only JSON, but sometimes adds markdown code fences
        json_text = _extract_json_from_response(response_text)
        
        extracted_fields = json.loads(json_text)
        
        # Validate the extracted fields
        validated_fields = _validate_extracted_fields(extracted_fields)
        
        # Log usage for cost tracking
        usage = response_body.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        logger.info(f"Claude extraction complete. Tokens: {input_tokens} in, {output_tokens} out")
        
        return True, validated_fields, None
        
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse Claude response as JSON: {str(e)}"
        logger.error(error_msg)
        return False, None, error_msg
        
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_msg = f"Bedrock API error: {error_code} - {str(e)}"
        logger.error(error_msg)
        return False, None, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error during Claude extraction: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, None, error_msg


def _extract_json_from_response(response_text: str) -> str:
    """
    Extract JSON from Claude's response, handling markdown code fences.
    
    Claude sometimes wraps JSON in markdown code blocks like:
    ```json
    {"field": "value"}
    ```
    
    This function strips those wrappers to get clean JSON.
    
    Args:
        response_text: Raw text from Claude's response.
    
    Returns:
        str: Clean JSON string ready for parsing.
    """
    text = response_text.strip()
    
    # Remove markdown code fences if present
    if text.startswith("```json"):
        text = text[7:]  # Remove ```json
    elif text.startswith("```"):
        text = text[3:]  # Remove ```
    
    if text.endswith("```"):
        text = text[:-3]  # Remove closing ```
    
    return text.strip()


def _validate_extracted_fields(fields: Dict) -> Dict:
    """
    Validate and normalize extracted fields.
    
    Ensures:
    - Numeric fields are proper numbers
    - Dates are in correct format
    - Unknown vehicle classes are normalized
    - Missing fields are explicitly null
    
    Args:
        fields: Raw extracted fields from Claude.
    
    Returns:
        dict: Validated and normalized fields.
    """
    validated = {}
    
    # String fields - just copy as-is or set to None
    string_fields = [
        "hire_start_date", "hire_end_date", "vehicle_class",
        "hire_company", "claimant_name", "vehicle_registration", "region"
    ]
    for field in string_fields:
        value = fields.get(field)
        validated[field] = value if value else None
    
    # Integer fields - convert to int
    int_fields = ["hire_days"]
    for field in int_fields:
        value = fields.get(field)
        if value is not None:
            try:
                validated[field] = int(value)
            except (ValueError, TypeError):
                validated[field] = None
        else:
            validated[field] = None
    
    # Float fields - convert to float
    float_fields = ["daily_rate", "total_claimed"]
    for field in float_fields:
        value = fields.get(field)
        if value is not None:
            try:
                # Handle strings with currency symbols
                if isinstance(value, str):
                    value = value.replace("£", "").replace(",", "").strip()
                validated[field] = float(value)
            except (ValueError, TypeError):
                validated[field] = None
        else:
            validated[field] = None
    
    # Normalize vehicle class to uppercase
    if validated.get("vehicle_class"):
        validated["vehicle_class"] = validated["vehicle_class"].upper().replace(" ", "_")
    
    # Normalize region to uppercase
    if validated.get("region"):
        validated["region"] = validated["region"].upper().replace(" ", "_")
    
    return validated


# =============================================================================
# Explanation Generation
# =============================================================================

def generate_claim_explanation(
    extracted_data: Dict,
    market_data: Dict,
    score_result: Dict
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Generate a human-readable explanation for a claim verdict.
    
    Uses Claude to create a clear, professional explanation of:
    - What was extracted from the document
    - How it compares to market rates
    - Why the verdict was reached
    - Key factors that influenced the decision
    
    This is crucial for governance - reviewers need to understand
    the AI's reasoning, not just accept its verdict.
    
    Args:
        extracted_data: Fields extracted from the document.
        market_data: Market rate comparison data.
        score_result: Scoring and verdict information.
    
    Returns:
        Tuple containing:
        - success (bool): True if explanation generated
        - explanation (str|None): Human-readable explanation
        - error (str|None): Error message if failed
    
    Example:
        >>> explanation = generate_claim_explanation(extracted, market, score)
        >>> print(explanation)
        "The claim for £89/day for a Group C vehicle in London was 
         classified as POTENTIALLY INFLATED. The market rate range 
         for this vehicle class is £45-65/day, giving an inflation 
         ratio of 1.37..."
    """
    # Build context for Claude
    context = f"""
Extracted Claim Data:
- Vehicle Class: {extracted_data.get('vehicle_class', 'Unknown')}
- Hire Period: {extracted_data.get('hire_days', 'Unknown')} days
- Daily Rate Claimed: £{extracted_data.get('daily_rate', 'Unknown')}
- Total Claimed: £{extracted_data.get('total_claimed', 'Unknown')}
- Region: {extracted_data.get('region', 'Unknown')}
- Hire Company: {extracted_data.get('hire_company', 'Unknown')}

Market Rate Data:
- Market Rate Range: £{market_data.get('market_rate_low', 'N/A')} - £{market_data.get('market_rate_high', 'N/A')} per day
- Source Year: {market_data.get('source_year', 'Unknown')}

Scoring Result:
- Verdict: {score_result.get('verdict', 'Unknown')}
- Inflation Ratio: {score_result.get('inflation_ratio', 'N/A')}
- Confidence Score: {score_result.get('confidence_score', 'N/A')}
"""

    prompt = f"""You are an expert insurance claims analyst. Write a clear, professional explanation of the following claim analysis. The explanation should:

1. Summarize what was claimed
2. Explain how it compares to market rates
3. Justify the verdict with specific numbers
4. Note any confidence concerns if applicable
5. Be suitable for a legal/compliance audience

Keep the explanation concise (3-5 sentences) but thorough.

{context}

Write the explanation:"""

    try:
        bedrock = get_bedrock_client()
        
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,  # Explanations should be concise
            "temperature": 0.3,  # Slight variation is OK for explanations
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        response = bedrock.invoke_model(
            modelId=settings.BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body)
        )
        
        response_body = json.loads(response["body"].read())
        content = response_body.get("content", [])
        
        if not content:
            return False, None, "Empty response from Claude"
        
        explanation = content[0].get("text", "").strip()
        
        logger.info("Generated claim explanation successfully")
        return True, explanation, None
        
    except Exception as e:
        error_msg = f"Failed to generate explanation: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, None, error_msg


# =============================================================================
# Document Classification
# =============================================================================

def classify_document_type(
    document_text: str
) -> Tuple[bool, Optional[str], Optional[float], Optional[str]]:
    """
    Classify the type of document being analyzed.
    
    Before processing, we check if the document is actually a
    credit hire invoice/court pack. This prevents wasted processing
    on irrelevant documents.
    
    Possible document types:
    - HIRE_INVOICE: Credit hire invoice (process this)
    - COURT_PACK: Legal bundle containing hire invoice (process this)
    - REPAIR_INVOICE: Vehicle repair invoice (different workflow)
    - POLICY_DOCUMENT: Insurance policy (not a claim)
    - OTHER: Unrelated document (reject)
    
    Args:
        document_text: Raw text from the document.
    
    Returns:
        Tuple containing:
        - success (bool): True if classification succeeded
        - doc_type (str|None): Document type classification
        - confidence (float|None): Confidence in classification (0-1)
        - error (str|None): Error message if failed
    """
    # Take just the first part of the document for classification
    # This is faster and cheaper than sending the whole thing
    sample_text = document_text[:5000] if len(document_text) > 5000 else document_text
    
    prompt = f"""Classify this document into one of these categories:
- HIRE_INVOICE: A credit hire or vehicle rental invoice
- COURT_PACK: A legal document bundle containing hire claim information
- REPAIR_INVOICE: A vehicle repair or bodyshop invoice
- POLICY_DOCUMENT: An insurance policy document
- OTHER: Any other type of document

Return ONLY a JSON object with "type" and "confidence" (0-1).

Document excerpt:
{sample_text}

Response (JSON only):"""

    try:
        bedrock = get_bedrock_client()
        
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        response = bedrock.invoke_model(
            modelId=settings.BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body)
        )
        
        response_body = json.loads(response["body"].read())
        content = response_body.get("content", [])
        
        if not content:
            return False, None, None, "Empty response"
        
        response_text = content[0].get("text", "")
        json_text = _extract_json_from_response(response_text)
        result = json.loads(json_text)
        
        doc_type = result.get("type", "OTHER")
        confidence = float(result.get("confidence", 0.5))
        
        logger.info(f"Document classified as {doc_type} (confidence: {confidence:.2f})")
        return True, doc_type, confidence, None
        
    except Exception as e:
        error_msg = f"Document classification failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, None, None, error_msg


# =============================================================================
# Logging and Audit
# =============================================================================

def log_ai_interaction(
    operation: str,
    input_summary: str,
    output_summary: str,
    tokens_used: Dict[str, int] = None
) -> None:
    """
    Log AI interactions for audit and governance.
    
    Every AI call should be logged to support:
    - Cost tracking (token usage)
    - Audit trail (what was asked, what was returned)
    - Debugging (when things go wrong)
    - Governance compliance
    
    Args:
        operation: Type of operation (extraction, explanation, classification)
        input_summary: Brief summary of input (not full text for privacy)
        output_summary: Brief summary of output
        tokens_used: Dict with input_tokens and output_tokens
    """
    log_entry = {
        "operation": operation,
        "input_summary": input_summary[:200],  # Truncate for logs
        "output_summary": output_summary[:200],
        "model": settings.BEDROCK_MODEL_ID,
        "tokens": tokens_used or {}
    }
    
    logger.info(f"AI Interaction: {json.dumps(log_entry)}")
