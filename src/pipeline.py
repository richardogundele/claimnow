# =============================================================================
# ClaimsNOW - Pipeline Orchestrator Module
# =============================================================================
# This module coordinates the full claim analysis workflow.
#
# The pipeline orchestrates all components in sequence:
# 1. Upload document to S3
# 2. Extract text with Textract
# 3. Parse fields with the Extractor (regex-based)
# 4. If extraction incomplete, escalate to Bedrock (AI-powered)
# 5. Match against market rates
# 6. Score and classify the claim
# 7. Generate explanation
# 8. Store result for audit trail
#
# This is the "agentic" design - deterministic methods first,
# AI fallback when needed for robust handling of any document.
# =============================================================================

import uuid
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple, Any, List

from src.config import settings, Verdict
from src.aws_s3 import upload_file, upload_result
from src.aws_textract import extract_document_content
from src.aws_dynamodb import store_claim_result, get_claim_result
from src.aws_bedrock import extract_fields_with_claude, classify_document_type
from src.extractor import extract_fields, validate_extraction
from src.rate_matcher import find_market_rate, compare_rate
from src.scorer import score_claim
from src.explainer import generate_explanation, generate_report

# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline Configuration
# =============================================================================

# Minimum confidence threshold for regex extraction
# Below this, we escalate to AI extraction
MIN_EXTRACTION_CONFIDENCE = 0.7

# Required fields that must be present for scoring
REQUIRED_FOR_SCORING = ["daily_rate", "vehicle_class"]


# =============================================================================
# Main Pipeline Function
# =============================================================================

def analyse_claim(
    file_bytes: bytes,
    filename: str,
    options: Dict[str, Any] = None
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Run the full claim analysis pipeline.
    
    This is the main entry point for analyzing a claim document.
    It orchestrates all stages of the pipeline and handles errors gracefully.
    
    Args:
        file_bytes: The PDF document content as bytes.
        filename: Original filename for reference.
        options: Optional configuration:
            - skip_ai: bool - Skip AI extraction fallback
            - force_ai: bool - Always use AI extraction
            - save_result: bool - Save to DynamoDB (default True)
            - generate_report: bool - Generate formatted report
    
    Returns:
        Tuple containing:
        - success (bool): True if analysis completed successfully
        - result (dict|None): Complete analysis result:
            - claim_id: Unique identifier for this analysis
            - filename: Original document filename
            - verdict: FAIR, POTENTIALLY_INFLATED, FLAGGED, or INSUFFICIENT_DATA
            - confidence_score: Overall confidence in the result
            - extracted_data: All extracted fields
            - market_comparison: Rate comparison data
            - scoring_result: Full scoring breakdown
            - explanation: Human-readable explanation
            - processing_time_ms: Total processing time
            - stages_completed: List of pipeline stages completed
        - error (str|None): Error message if failed
    
    Example:
        >>> with open("claim.pdf", "rb") as f:
        ...     success, result, error = analyse_claim(f.read(), "claim.pdf")
        >>> if success:
        ...     print(f"Verdict: {result['verdict']}")
    """
    # Initialize timing and tracking
    start_time = datetime.utcnow()
    stages_completed = []
    options = options or {}
    
    # Generate unique claim ID
    claim_id = str(uuid.uuid4())
    
    logger.info(f"Starting claim analysis: {claim_id} ({filename})")
    
    try:
        # =====================================================================
        # STAGE 1: Upload Document to S3
        # =====================================================================
        logger.info(f"[{claim_id}] Stage 1: Uploading document to S3")
        
        success, s3_key, error = upload_file(file_bytes, filename)
        
        if not success:
            return False, None, f"Failed to upload document: {error}"
        
        stages_completed.append({
            "stage": "upload",
            "status": "success",
            "s3_key": s3_key
        })
        
        # =====================================================================
        # STAGE 2: Extract Text with Textract
        # =====================================================================
        logger.info(f"[{claim_id}] Stage 2: Extracting text with Textract")
        
        success, textract_content, error = extract_document_content(
            settings.S3_BUCKET_NAME,
            s3_key,
            include_tables=True
        )
        
        if not success:
            return False, None, f"Failed to extract text: {error}"
        
        raw_text = textract_content.get("text", "")
        tables = textract_content.get("tables", [])
        key_value_pairs = textract_content.get("key_value_pairs", {})
        page_count = textract_content.get("page_count", 0)
        
        stages_completed.append({
            "stage": "textract",
            "status": "success",
            "page_count": page_count,
            "table_count": len(tables),
            "text_length": len(raw_text)
        })
        
        # =====================================================================
        # STAGE 3: Extract Fields (Regex-based)
        # =====================================================================
        logger.info(f"[{claim_id}] Stage 3: Extracting fields with regex")
        
        extracted_data, extraction_confidence, missing_fields = extract_fields(
            raw_text,
            tables,
            key_value_pairs
        )
        
        stages_completed.append({
            "stage": "extraction_regex",
            "status": "success",
            "confidence": extraction_confidence,
            "missing_fields": missing_fields
        })
        
        # =====================================================================
        # STAGE 4: AI Extraction Fallback (if needed)
        # =====================================================================
        use_ai = _should_use_ai_extraction(
            extraction_confidence,
            missing_fields,
            options
        )
        
        if use_ai:
            logger.info(f"[{claim_id}] Stage 4: AI extraction fallback")
            
            success, ai_extracted, error = extract_fields_with_claude(
                raw_text,
                partial_data=extracted_data
            )
            
            if success and ai_extracted:
                # Merge AI results with regex results (AI fills gaps)
                extracted_data = _merge_extracted_data(extracted_data, ai_extracted)
                
                # Recalculate confidence with merged data
                extraction_confidence = _calculate_merged_confidence(extracted_data)
                
                stages_completed.append({
                    "stage": "extraction_ai",
                    "status": "success",
                    "fields_filled": len([k for k, v in ai_extracted.items() if v])
                })
            else:
                logger.warning(f"[{claim_id}] AI extraction failed: {error}")
                stages_completed.append({
                    "stage": "extraction_ai",
                    "status": "failed",
                    "error": error
                })
        else:
            logger.info(f"[{claim_id}] Skipping AI extraction (confidence sufficient)")
            stages_completed.append({
                "stage": "extraction_ai",
                "status": "skipped",
                "reason": "Confidence sufficient or AI disabled"
            })
        
        # Validate extracted data
        is_valid, validation_warnings = validate_extraction(extracted_data)
        
        if validation_warnings:
            logger.warning(f"[{claim_id}] Validation warnings: {validation_warnings}")
        
        # =====================================================================
        # STAGE 5: Market Rate Matching
        # =====================================================================
        logger.info(f"[{claim_id}] Stage 5: Finding market rates")
        
        vehicle_class = extracted_data.get("vehicle_class")
        region = extracted_data.get("region")
        hire_days = extracted_data.get("hire_days")
        
        success, market_data, error = find_market_rate(
            vehicle_class or "UNKNOWN",
            region,
            hire_days
        )
        
        if not success:
            # Use empty market data - scorer will handle this
            market_data = {
                "market_rate_low": None,
                "market_rate_high": None,
                "source": "unavailable"
            }
        
        stages_completed.append({
            "stage": "rate_matching",
            "status": "success" if success else "partial",
            "market_rate_found": success,
            "source": market_data.get("source")
        })
        
        # =====================================================================
        # STAGE 6: Score and Classify
        # =====================================================================
        logger.info(f"[{claim_id}] Stage 6: Scoring claim")
        
        scoring_result = score_claim(
            extracted_data,
            market_data,
            extraction_confidence
        )
        
        verdict = scoring_result.get("verdict", Verdict.INSUFFICIENT_DATA)
        
        stages_completed.append({
            "stage": "scoring",
            "status": "success",
            "verdict": verdict,
            "inflation_ratio": scoring_result.get("inflation_ratio")
        })
        
        # =====================================================================
        # STAGE 7: Generate Explanation
        # =====================================================================
        logger.info(f"[{claim_id}] Stage 7: Generating explanation")
        
        explanation = generate_explanation(
            extracted_data,
            market_data,
            scoring_result
        )
        
        stages_completed.append({
            "stage": "explanation",
            "status": "success"
        })
        
        # Generate formatted report if requested
        formatted_report = None
        if options.get("generate_report"):
            formatted_report = generate_report(
                extracted_data,
                market_data,
                scoring_result,
                format=options.get("report_format", "text")
            )
        
        # =====================================================================
        # STAGE 8: Store Result
        # =====================================================================
        if options.get("save_result", True):
            logger.info(f"[{claim_id}] Stage 8: Storing result")
            
            result_data = {
                "filename": filename,
                "s3_key": s3_key,
                "verdict": verdict,
                "confidence_score": scoring_result.get("confidence_score"),
                "inflation_ratio": scoring_result.get("inflation_ratio"),
                "extracted_data": extracted_data,
                "market_data": market_data,
                "explanation_summary": explanation.get("summary")
            }
            
            success, stored_claim_id, error = store_claim_result(result_data)
            
            if success:
                # Use the stored claim ID (might differ from our generated one)
                claim_id = stored_claim_id
                
            stages_completed.append({
                "stage": "storage",
                "status": "success" if success else "failed",
                "claim_id": claim_id
            })
        else:
            stages_completed.append({
                "stage": "storage",
                "status": "skipped"
            })
        
        # =====================================================================
        # Build Final Result
        # =====================================================================
        end_time = datetime.utcnow()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        result = {
            # Identifiers
            "claim_id": claim_id,
            "filename": filename,
            "s3_key": s3_key,
            
            # Core verdict
            "verdict": verdict,
            "confidence_score": scoring_result.get("confidence_score"),
            "inflation_ratio": scoring_result.get("inflation_ratio"),
            
            # Detailed data
            "extracted_data": extracted_data,
            "market_comparison": {
                "claimed_rate": extracted_data.get("daily_rate"),
                "market_rate_low": market_data.get("market_rate_low"),
                "market_rate_high": market_data.get("market_rate_high"),
                "rate_source": market_data.get("source")
            },
            "scoring_result": scoring_result,
            "explanation": explanation,
            
            # Optional report
            "formatted_report": formatted_report,
            
            # Validation
            "validation_warnings": validation_warnings,
            
            # Metadata
            "processing_time_ms": processing_time_ms,
            "stages_completed": stages_completed,
            "analysed_at": end_time.isoformat()
        }
        
        logger.info(f"[{claim_id}] Analysis complete: {verdict} in {processing_time_ms}ms")
        
        return True, result, None
        
    except Exception as e:
        # Catch-all for unexpected errors
        error_msg = f"Pipeline error: {str(e)}"
        logger.error(f"[{claim_id}] {error_msg}", exc_info=True)
        
        return False, None, error_msg


# =============================================================================
# Helper Functions
# =============================================================================

def _should_use_ai_extraction(
    confidence: float,
    missing_fields: List[str],
    options: Dict[str, Any]
) -> bool:
    """
    Determine whether to use AI extraction fallback.
    
    AI extraction is used when:
    - Confidence is below threshold, OR
    - Required fields are missing, AND
    - AI extraction is not disabled
    
    Args:
        confidence: Current extraction confidence.
        missing_fields: List of fields not found.
        options: Pipeline options.
    
    Returns:
        bool: True if AI extraction should be used.
    """
    # Check if AI is explicitly disabled
    if options.get("skip_ai"):
        return False
    
    # Check if AI is forced
    if options.get("force_ai"):
        return True
    
    # Check confidence threshold
    if confidence < MIN_EXTRACTION_CONFIDENCE:
        return True
    
    # Check for missing required fields
    for required_field in REQUIRED_FOR_SCORING:
        if required_field in missing_fields:
            return True
    
    return False


def _merge_extracted_data(
    regex_data: Dict[str, Any],
    ai_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge regex and AI extracted data.
    
    Strategy:
    - Keep regex values where they exist (more reliable)
    - Fill gaps with AI values
    - Prefer AI for complex fields if regex seems wrong
    
    Args:
        regex_data: Data extracted with regex.
        ai_data: Data extracted with AI.
    
    Returns:
        dict: Merged extracted data.
    """
    merged = regex_data.copy()
    
    for key, ai_value in ai_data.items():
        regex_value = merged.get(key)
        
        # If regex didn't find the value, use AI
        if regex_value is None and ai_value is not None:
            merged[key] = ai_value
            logger.debug(f"AI filled missing field: {key} = {ai_value}")
        
        # For certain fields, prefer AI if regex seems incorrect
        # (e.g., vehicle_class when regex returned UNKNOWN)
        if key == "vehicle_class":
            if regex_value == "UNKNOWN" and ai_value and ai_value != "UNKNOWN":
                merged[key] = ai_value
                logger.debug(f"AI improved vehicle_class: {regex_value} -> {ai_value}")
    
    return merged


def _calculate_merged_confidence(extracted_data: Dict[str, Any]) -> float:
    """
    Calculate confidence score for merged data.
    
    Args:
        extracted_data: Merged extracted data.
    
    Returns:
        float: Confidence score (0.0-1.0)
    """
    # Use the same weights as in extractor
    confidence = 0.0
    
    if extracted_data.get("hire_start_date") and extracted_data.get("hire_end_date"):
        confidence += settings.CONFIDENCE_WEIGHT_HIRE_DATES
    elif extracted_data.get("hire_days"):
        confidence += settings.CONFIDENCE_WEIGHT_HIRE_DATES * 0.5
    
    if extracted_data.get("vehicle_class"):
        confidence += settings.CONFIDENCE_WEIGHT_VEHICLE_CLASS
    
    if extracted_data.get("daily_rate"):
        confidence += settings.CONFIDENCE_WEIGHT_DAILY_RATE
    
    if extracted_data.get("total_claimed"):
        confidence += settings.CONFIDENCE_WEIGHT_TOTAL_CLAIMED
    
    if extracted_data.get("hire_company"):
        confidence += settings.CONFIDENCE_WEIGHT_HIRE_COMPANY
    
    return min(confidence, 1.0)


# =============================================================================
# Batch Processing
# =============================================================================

def analyse_batch(
    documents: List[Tuple[bytes, str]],
    options: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Process multiple documents in a batch.
    
    Args:
        documents: List of (file_bytes, filename) tuples.
        options: Pipeline options applied to all documents.
    
    Returns:
        dict: Batch results:
            - total: Number of documents processed
            - successful: Number successfully analyzed
            - failed: Number that failed
            - results: List of individual results
            - summary: Aggregate statistics
    """
    options = options or {}
    results = []
    successful = 0
    failed = 0
    
    logger.info(f"Starting batch analysis of {len(documents)} documents")
    
    for i, (file_bytes, filename) in enumerate(documents):
        logger.info(f"Processing document {i+1}/{len(documents)}: {filename}")
        
        try:
            success, result, error = analyse_claim(file_bytes, filename, options)
            
            if success:
                results.append({
                    "index": i,
                    "filename": filename,
                    "success": True,
                    "result": result
                })
                successful += 1
            else:
                results.append({
                    "index": i,
                    "filename": filename,
                    "success": False,
                    "error": error
                })
                failed += 1
                
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            results.append({
                "index": i,
                "filename": filename,
                "success": False,
                "error": str(e)
            })
            failed += 1
    
    # Generate summary statistics
    verdicts = {}
    total_confidence = 0
    confidence_count = 0
    
    for r in results:
        if r["success"]:
            verdict = r["result"]["verdict"]
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
            
            confidence = r["result"].get("confidence_score", 0)
            total_confidence += confidence
            confidence_count += 1
    
    summary = {
        "total": len(documents),
        "successful": successful,
        "failed": failed,
        "verdicts": verdicts,
        "average_confidence": total_confidence / confidence_count if confidence_count > 0 else 0
    }
    
    logger.info(f"Batch complete: {successful}/{len(documents)} successful")
    
    return {
        "total": len(documents),
        "successful": successful,
        "failed": failed,
        "results": results,
        "summary": summary
    }


# =============================================================================
# Retrieval Functions
# =============================================================================

def get_analysis(claim_id: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Retrieve a previous claim analysis.
    
    Args:
        claim_id: The unique ID of the claim to retrieve.
    
    Returns:
        Tuple containing:
        - success (bool): True if claim was found
        - result (dict|None): The stored claim data
        - error (str|None): Error message if failed
    """
    return get_claim_result(claim_id)


# =============================================================================
# Validation Pipeline
# =============================================================================

def validate_document(
    file_bytes: bytes,
    filename: str
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Validate a document before full analysis.
    
    Checks:
    - Document is readable
    - Document appears to be a hire invoice
    - Basic data can be extracted
    
    Args:
        file_bytes: The document content.
        filename: Original filename.
    
    Returns:
        Tuple containing:
        - valid (bool): True if document is valid for processing
        - info (dict): Document information and any warnings
        - error (str|None): Error message if invalid
    """
    info = {
        "filename": filename,
        "file_size_kb": len(file_bytes) / 1024,
        "warnings": []
    }
    
    # Check file size
    if len(file_bytes) > 50 * 1024 * 1024:  # 50MB limit
        return False, info, "File exceeds 50MB size limit"
    
    if len(file_bytes) < 1024:  # Less than 1KB
        info["warnings"].append("File is very small, may be empty or corrupted")
    
    # Check file type (basic check)
    if not filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg', '.tiff')):
        return False, info, "Unsupported file type. Please upload PDF or image files."
    
    # Could add more validation here:
    # - PDF parsing check
    # - Document classification (is it actually a hire invoice?)
    
    info["valid_format"] = True
    
    return True, info, None


# =============================================================================
# Pipeline Status
# =============================================================================

def get_pipeline_status() -> Dict[str, Any]:
    """
    Get the current status of the pipeline and its components.
    
    Useful for health checks and monitoring.
    
    Returns:
        dict: Pipeline status information.
    """
    status = {
        "pipeline_version": "1.0.0",
        "components": {},
        "configuration": {
            "min_extraction_confidence": MIN_EXTRACTION_CONFIDENCE,
            "threshold_fair": settings.THRESHOLD_FAIR,
            "threshold_inflated": settings.THRESHOLD_INFLATED
        }
    }
    
    # Check S3
    try:
        from src.aws_s3 import get_s3_client
        get_s3_client()
        status["components"]["s3"] = {"status": "available"}
    except Exception as e:
        status["components"]["s3"] = {"status": "unavailable", "error": str(e)}
    
    # Check Textract
    try:
        from src.aws_textract import get_textract_client
        get_textract_client()
        status["components"]["textract"] = {"status": "available"}
    except Exception as e:
        status["components"]["textract"] = {"status": "unavailable", "error": str(e)}
    
    # Check DynamoDB
    try:
        from src.aws_dynamodb import get_dynamodb_resource
        get_dynamodb_resource()
        status["components"]["dynamodb"] = {"status": "available"}
    except Exception as e:
        status["components"]["dynamodb"] = {"status": "unavailable", "error": str(e)}
    
    # Check Bedrock
    try:
        from src.aws_bedrock import get_bedrock_client
        get_bedrock_client()
        status["components"]["bedrock"] = {"status": "available"}
    except Exception as e:
        status["components"]["bedrock"] = {"status": "unavailable", "error": str(e)}
    
    # Overall status
    all_available = all(
        c.get("status") == "available" 
        for c in status["components"].values()
    )
    status["overall_status"] = "healthy" if all_available else "degraded"
    
    return status
