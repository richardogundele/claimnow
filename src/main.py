# =============================================================================
# ClaimsNOW - FastAPI Application Entry Point
# =============================================================================
# This module defines the REST API for the ClaimsNOW application.
#
# API Endpoints:
# - POST /analyse: Upload a PDF and get claim analysis
# - GET /claims/{claim_id}: Retrieve a previous analysis
# - GET /claims: List recent claim analyses
# - GET /health: Health check endpoint
# - GET /status: Detailed system status
#
# The API is designed to be:
# - RESTful with clear resource-based URLs
# - Well-documented with OpenAPI/Swagger
# - CORS-enabled for frontend integration
# - Error-handled with informative messages
# =============================================================================

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import settings, get_cors_origins_list, Verdict
from src.pipeline import (
    analyse_claim,
    get_analysis,
    validate_document,
    get_pipeline_status,
    analyse_batch
)
from src.aws_dynamodb import list_recent_claims, update_claim_status, seed_sample_rates

# =============================================================================
# Logging Setup
# =============================================================================

# Configure logging for the application
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# FastAPI Application
# =============================================================================

# Create the FastAPI application with metadata for documentation
app = FastAPI(
    title="ClaimsNOW API",
    description="""
    ## AI-Powered Motor Insurance Court Pack Analyser
    
    ClaimsNOW analyzes credit hire invoices and court pack documents to:
    - Extract key claim data (rates, dates, vehicle info)
    - Compare against market rate benchmarks
    - Classify claims as Fair, Potentially Inflated, or Flagged
    - Generate explainable verdicts for human review
    
    ### Key Features
    - **Document AI**: PDF text extraction with AWS Textract
    - **NLP Extraction**: Hybrid regex + Claude AI field extraction
    - **Rate Comparison**: Match against reference market rates
    - **Explainability**: Every verdict comes with full reasoning
    
    ### Authentication
    Currently the API is open for development. Production deployments
    should implement API key or OAuth authentication.
    """,
    version="1.0.0",
    contact={
        "name": "Richard Ademola Ogundele",
        "email": "Ogundelerichard27@gmail.com"
    },
    license_info={
        "name": "Proprietary",
    }
)

# =============================================================================
# CORS Middleware
# =============================================================================

# Enable CORS for frontend access
# This allows the React frontend to call the API
app.add_middleware(
    CORSMiddleware,
    # Allow requests from configured origins
    allow_origins=get_cors_origins_list(),
    # Allow credentials (cookies, authorization headers)
    allow_credentials=True,
    # Allow all HTTP methods
    allow_methods=["*"],
    # Allow all headers
    allow_headers=["*"],
)


# =============================================================================
# Pydantic Models - Request/Response Schemas
# =============================================================================

class AnalysisOptions(BaseModel):
    """
    Options for claim analysis.
    
    These control how the analysis pipeline behaves.
    """
    skip_ai: bool = Field(
        default=False,
        description="Skip AI-powered extraction (use regex only)"
    )
    force_ai: bool = Field(
        default=False,
        description="Always use AI extraction regardless of regex confidence"
    )
    generate_report: bool = Field(
        default=False,
        description="Generate a formatted text/HTML report"
    )
    report_format: str = Field(
        default="text",
        description="Report format: 'text' or 'html'"
    )


class ClaimSummary(BaseModel):
    """
    Brief summary of a claim analysis.
    
    Used in list responses where full details aren't needed.
    """
    claim_id: str = Field(description="Unique identifier for the claim")
    created_at: str = Field(description="When the analysis was performed")
    verdict: str = Field(description="Analysis verdict")
    confidence_score: Optional[float] = Field(description="Confidence in the verdict")


class ExtractedData(BaseModel):
    """
    Data extracted from the claim document.
    """
    daily_rate: Optional[float] = Field(description="Daily hire rate in GBP")
    total_claimed: Optional[float] = Field(description="Total amount claimed in GBP")
    hire_days: Optional[int] = Field(description="Number of hire days")
    vehicle_class: Optional[str] = Field(description="Vehicle category/group")
    region: Optional[str] = Field(description="UK region")
    hire_company: Optional[str] = Field(description="Credit hire company name")
    hire_start_date: Optional[str] = Field(description="Start date of hire (YYYY-MM-DD)")
    hire_end_date: Optional[str] = Field(description="End date of hire (YYYY-MM-DD)")
    claimant_name: Optional[str] = Field(description="Name of claimant")
    vehicle_registration: Optional[str] = Field(description="Vehicle registration number")


class MarketComparison(BaseModel):
    """
    Market rate comparison data.
    """
    claimed_rate: Optional[float] = Field(description="The claimed daily rate")
    market_rate_low: Optional[float] = Field(description="Lower bound of market rate")
    market_rate_high: Optional[float] = Field(description="Upper bound of market rate")
    rate_source: Optional[str] = Field(description="Source of market rate data")


class AnalysisResult(BaseModel):
    """
    Complete claim analysis result.
    """
    claim_id: str = Field(description="Unique identifier for this analysis")
    filename: str = Field(description="Original document filename")
    verdict: str = Field(description="FAIR, POTENTIALLY_INFLATED, FLAGGED, or INSUFFICIENT_DATA")
    confidence_score: float = Field(description="Overall confidence in the result (0-1)")
    inflation_ratio: Optional[float] = Field(description="Claimed rate / Market rate high")
    extracted_data: ExtractedData = Field(description="Data extracted from document")
    market_comparison: MarketComparison = Field(description="Rate comparison data")
    processing_time_ms: int = Field(description="Total processing time in milliseconds")
    analysed_at: str = Field(description="Timestamp of analysis")


class HealthResponse(BaseModel):
    """
    Health check response.
    """
    status: str = Field(description="Overall health status")
    timestamp: str = Field(description="Current server time")
    version: str = Field(description="API version")


class StatusResponse(BaseModel):
    """
    Detailed system status response.
    """
    pipeline_version: str
    overall_status: str
    components: Dict[str, Any]
    configuration: Dict[str, Any]


class ReviewUpdate(BaseModel):
    """
    Request body for updating claim review status.
    """
    status: str = Field(
        description="New status: reviewed, approved, disputed, or closed"
    )
    reviewer_notes: Optional[str] = Field(
        default=None,
        description="Optional notes from the reviewer"
    )


class ErrorResponse(BaseModel):
    """
    Standard error response format.
    """
    error: str = Field(description="Error type")
    message: str = Field(description="Detailed error message")
    claim_id: Optional[str] = Field(default=None, description="Related claim ID if applicable")


# =============================================================================
# API Endpoints
# =============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health Check",
    description="Check if the API is running. Returns basic health status."
)
async def health_check():
    """
    Simple health check endpoint.
    
    Returns 200 OK if the API is running.
    Used by load balancers and monitoring systems.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


@app.get(
    "/status",
    response_model=StatusResponse,
    tags=["System"],
    summary="System Status",
    description="Get detailed status of all system components."
)
async def system_status():
    """
    Detailed system status including all pipeline components.
    
    Checks:
    - S3 connectivity
    - Textract availability
    - DynamoDB connectivity
    - Bedrock availability
    """
    status = get_pipeline_status()
    return status


@app.post(
    "/analyse",
    response_model=AnalysisResult,
    tags=["Analysis"],
    summary="Analyse a Claim Document",
    description="""
    Upload a PDF court pack or invoice document for analysis.
    
    The system will:
    1. Extract text from the document
    2. Identify key fields (rates, dates, vehicle info)
    3. Compare against market rate benchmarks
    4. Return a verdict with full explanation
    
    **Supported formats:** PDF, PNG, JPG, TIFF
    **Max file size:** 50MB
    """,
    responses={
        200: {"description": "Analysis completed successfully"},
        400: {"description": "Invalid document or request"},
        500: {"description": "Server error during processing"}
    }
)
async def analyse_document(
    file: UploadFile = File(..., description="The PDF document to analyse"),
    skip_ai: bool = Query(
        default=False,
        description="Skip AI extraction (use regex only)"
    ),
    force_ai: bool = Query(
        default=False,
        description="Always use AI extraction"
    )
):
    """
    Analyse a claim document and return verdict.
    
    This is the main analysis endpoint. It runs the full pipeline:
    - Upload to S3
    - Extract text with Textract
    - Parse fields with regex + AI
    - Compare to market rates
    - Score and classify
    - Generate explanation
    """
    # Read file content
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded file: {str(e)}"
        )
    
    # Validate document
    is_valid, info, error = validate_document(file_bytes, file.filename)
    
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=error
        )
    
    # Log any warnings
    for warning in info.get("warnings", []):
        logger.warning(f"Document warning: {warning}")
    
    # Run analysis pipeline
    options = {
        "skip_ai": skip_ai,
        "force_ai": force_ai,
        "save_result": True
    }
    
    success, result, error = analyse_claim(file_bytes, file.filename, options)
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {error}"
        )
    
    # Return the analysis result
    return result


@app.get(
    "/claims/{claim_id}",
    response_model=AnalysisResult,
    tags=["Claims"],
    summary="Get Claim Analysis",
    description="Retrieve a previously completed claim analysis by its ID.",
    responses={
        200: {"description": "Claim found and returned"},
        404: {"description": "Claim not found"}
    }
)
async def get_claim(claim_id: str):
    """
    Retrieve a specific claim analysis by ID.
    
    Returns the full analysis result including:
    - Extracted data
    - Market comparison
    - Verdict and explanation
    """
    success, claim, error = get_analysis(claim_id)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Claim not found: {claim_id}"
        )
    
    return claim


@app.get(
    "/claims",
    response_model=List[ClaimSummary],
    tags=["Claims"],
    summary="List Recent Claims",
    description="Get a list of recent claim analyses.",
)
async def list_claims(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of claims to return"
    )
):
    """
    List recent claim analyses.
    
    Returns summary information for each claim.
    Use GET /claims/{id} to get full details.
    """
    success, claims, error = list_recent_claims(limit)
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list claims: {error}"
        )
    
    return claims


@app.patch(
    "/claims/{claim_id}/review",
    tags=["Claims"],
    summary="Update Claim Review Status",
    description="Update the review status of a claim after human review.",
    responses={
        200: {"description": "Status updated successfully"},
        404: {"description": "Claim not found"}
    }
)
async def update_review_status(
    claim_id: str,
    update: ReviewUpdate
):
    """
    Update the review status of a claim.
    
    Valid statuses:
    - reviewed: Human has checked the analysis
    - approved: Claim accepted as-is
    - disputed: Claim is under dispute
    - closed: Claim processing complete
    """
    # Verify claim exists
    success, claim, _ = get_analysis(claim_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Claim not found: {claim_id}"
        )
    
    # Update status
    success, error = update_claim_status(
        claim_id,
        update.status,
        update.reviewer_notes
    )
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update status: {error}"
        )
    
    return {"message": f"Claim {claim_id} status updated to {update.status}"}


@app.post(
    "/validate",
    tags=["Analysis"],
    summary="Validate Document",
    description="Validate a document before full analysis. Checks format and readability."
)
async def validate_upload(
    file: UploadFile = File(..., description="The document to validate")
):
    """
    Pre-validate a document before analysis.
    
    Checks:
    - File format is supported
    - File size is within limits
    - File appears to be readable
    
    Use this for quick validation before committing to full analysis.
    """
    file_bytes = await file.read()
    
    is_valid, info, error = validate_document(file_bytes, file.filename)
    
    return {
        "valid": is_valid,
        "filename": file.filename,
        "file_size_kb": info.get("file_size_kb"),
        "warnings": info.get("warnings", []),
        "error": error
    }


# =============================================================================
# Admin Endpoints
# =============================================================================

@app.post(
    "/admin/seed-rates",
    tags=["Admin"],
    summary="Seed Sample Market Rates",
    description="Populate the market rates database with sample data for testing."
)
async def seed_rates():
    """
    Seed the market rates database with sample data.
    
    This creates sample market rates for testing purposes.
    In production, rates would be populated from the main rate database.
    """
    success, count, error = seed_sample_rates()
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to seed rates: {error}"
        )
    
    return {
        "message": f"Successfully seeded {count} market rate records",
        "count": count
    }


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Custom handler for HTTP exceptions.
    
    Ensures consistent error response format.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "request_error",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    Catch-all handler for unexpected exceptions.
    
    Logs the error and returns a generic message
    (avoids leaking internal details).
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred. Please try again.",
            "status_code": 500
        }
    )


# =============================================================================
# Startup and Shutdown Events
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Actions to perform when the API starts.
    """
    logger.info("ClaimsNOW API starting up...")
    logger.info(f"CORS origins: {get_cors_origins_list()}")
    logger.info(f"AWS Region: {settings.AWS_REGION}")
    logger.info("API ready to accept requests")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Actions to perform when the API shuts down.
    """
    logger.info("ClaimsNOW API shutting down...")


# =============================================================================
# Run Server (for local development)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run the server with uvicorn
    # In production, use: uvicorn src.main:app --host 0.0.0.0 --port 8000
    uvicorn.run(
        "src.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_DEBUG,  # Hot reload in debug mode
        log_level="info"
    )
