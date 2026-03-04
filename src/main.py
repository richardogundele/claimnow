"""
main.py - FastAPI Application for ClaimsNOW

WHY THIS FILE EXISTS:
- Expose the claim analysis pipeline via REST API
- Handle file uploads from frontend
- Provide endpoints for all functionality
- Serve as the entry point for the application

FASTAPI CONCEPTS:
- Endpoints: URL paths that handle requests (GET, POST, etc.)
- Request Body: JSON data sent to the API
- File Upload: Handle PDF file uploads
- Response Model: Define the structure of responses
- Dependency Injection: Share resources across endpoints

RUNNING THE API:
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

API DOCUMENTATION:
    Once running, visit http://localhost:8000/docs for Swagger UI
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import from sibling modules using relative imports
# The '.' means "current package" - required when running as a module
from src.config import settings, ensure_directories
from src.pipeline import ClaimsPipeline, AnalysisResult, analyze_claim, analyze_claim_text
from src.rag_pipeline import RAGPipeline, rag_query
from src.vector_store import get_rates_store
from src.llm_client import get_llm_client

# Set up logging
logging.basicConfig(
    level=logging.INFO if settings.debug else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure required directories exist
ensure_directories()


# -----------------------------------------------------------------------------
# Pydantic Models (Request/Response Schemas)
# -----------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = "healthy"
    version: str = "2.0.0"
    bedrock_available: bool = False
    vectorstore_count: int = 0
    timestamp: str = ""


class AnalyzeTextRequest(BaseModel):
    """Request model for analyzing text directly."""
    text: str = Field(..., description="Document text to analyze")
    claim_id: Optional[str] = Field(None, description="Optional claim ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "CREDIT HIRE INVOICE\nVehicle: BMW 320d (Group D)\nDaily Rate: £65.00\nHire Period: 14 days",
                "claim_id": "TEST-001"
            }
        }


class RAGQueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str = Field(..., description="Question to ask")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Is £65/day fair for a Group C vehicle in London?",
                "filters": {"vehicle_group": "C"}
            }
        }


class AddRatesRequest(BaseModel):
    """Request model for adding rates to the vector store."""
    rates: List[Dict[str, Any]] = Field(..., description="List of rate records")
    
    class Config:
        json_schema_extra = {
            "example": {
                "rates": [
                    {
                        "text": "Group C vehicle hire in London, £52/day, 2024",
                        "vehicle_group": "C",
                        "region": "London",
                        "daily_rate": 52.0,
                        "year": 2024
                    }
                ]
            }
        }


class MarketSummaryResponse(BaseModel):
    """Response model for market summary."""
    vehicle_group: str
    region: Optional[str]
    statistics: Dict[str, Any]
    sample_rates: List[Dict[str, Any]]


# -----------------------------------------------------------------------------
# FastAPI Application
# -----------------------------------------------------------------------------
app = FastAPI(
    title="ClaimsNOW API",
    description="""
    AI-Powered Motor Insurance Court Pack Analyser
    
    This API provides endpoints for:
    - Analyzing claim documents (PDF upload or text)
    - Querying market rates using RAG
    - Managing the rate database
    
    Built with AWS Bedrock (Claude/Titan) + ChromaDB + scikit-learn
    """,
    version="2.0.0",
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc"     # ReDoc alternative
)

# Configure CORS (Cross-Origin Resource Sharing)
# This allows the React frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Initialize components on startup
# -----------------------------------------------------------------------------
pipeline: Optional[ClaimsPipeline] = None
rag: Optional[RAGPipeline] = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize components when the server starts.
    
    This runs once before the server accepts requests.
    """
    global pipeline, rag
    
    logger.info("Starting ClaimsNOW API...")
    
    # Initialize pipeline (this loads models)
    pipeline = ClaimsPipeline(verbose=settings.debug)
    
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Check Bedrock availability
    llm = get_llm_client()
    if llm.is_available():
        logger.info(f"Bedrock is available with model: {settings.bedrock_llm_model_id}")
    else:
        logger.warning("Bedrock is not available - LLM features will fail")
    
    # Log vector store status
    store = get_rates_store()
    logger.info(f"Vector store has {store.count} rate records")
    
    logger.info("ClaimsNOW API started successfully")


# -----------------------------------------------------------------------------
# Health Check Endpoint
# -----------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check API health and component status.
    
    Use this to verify the API is running and services are available.
    """
    llm = get_llm_client()
    store = get_rates_store()
    
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        bedrock_available=llm.is_available(),
        vectorstore_count=store.count,
        timestamp=datetime.now().isoformat()
    )


# -----------------------------------------------------------------------------
# Claim Analysis Endpoints
# -----------------------------------------------------------------------------
@app.post("/analyze/upload", tags=["Analysis"])
async def analyze_uploaded_file(
    file: UploadFile = File(..., description="PDF file to analyze"),
    claim_id: Optional[str] = Query(None, description="Optional claim ID")
):
    """
    Analyze an uploaded PDF document.
    
    This is the main endpoint for processing court pack documents.
    
    Steps:
    1. Upload PDF file
    2. Parse text from PDF
    3. Extract claim fields using LLM
    4. Match rates against market data
    5. Score the claim
    6. Return verdict with explanation
    
    Returns full analysis result including verdict, confidence, and explanation.
    """
    global pipeline
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    try:
        # Read file bytes
        contents = await file.read()
        
        # Analyze using pipeline
        result = pipeline.analyze_bytes(
            pdf_bytes=contents,
            filename=file.filename,
            claim_id=claim_id
        )
        
        return JSONResponse(content=result.to_dict())
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/analyze/text", tags=["Analysis"])
async def analyze_text(request: AnalyzeTextRequest):
    """
    Analyze claim from raw text.
    
    Use this when you already have extracted text (e.g., from testing).
    Skips the PDF parsing stage.
    """
    global pipeline
    
    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )
    
    try:
        result = pipeline.analyze_text(
            document_text=request.text,
            claim_id=request.claim_id
        )
        
        return JSONResponse(content=result.to_dict())
        
    except Exception as e:
        logger.error(f"Text analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


# -----------------------------------------------------------------------------
# RAG Query Endpoints
# -----------------------------------------------------------------------------
@app.post("/rag/query", tags=["RAG"])
async def rag_query_endpoint(request: RAGQueryRequest):
    """
    Query the RAG pipeline with a question.
    
    The RAG pipeline:
    1. Searches for relevant market rates
    2. Passes them as context to the LLM
    3. LLM generates an informed answer
    
    Example queries:
    - "Is £65/day fair for a Group C vehicle in London?"
    - "What are typical rates for luxury cars in Manchester?"
    """
    global rag
    
    try:
        response = rag.query(
            query=request.query,
            filters=request.filters
        )
        
        return JSONResponse(content={
            "answer": response.answer,
            "sources": response.sources,
            "success": response.success,
            "error": response.error
        })
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


# -----------------------------------------------------------------------------
# Rate Management Endpoints
# -----------------------------------------------------------------------------
@app.post("/rates/add", tags=["Rates"])
async def add_rates(request: AddRatesRequest):
    """
    Add rate records to the vector store.
    
    Each rate should include:
    - text: Description of the rate (used for embeddings)
    - vehicle_group: A-I
    - region: Geographic region
    - daily_rate: Rate in GBP
    - year: Year of the rate
    
    These rates are stored as embeddings and used for RAG queries.
    """
    store = get_rates_store()
    
    try:
        documents = []
        metadatas = []
        
        for rate in request.rates:
            # Text is required for embedding
            text = rate.get("text", "")
            if not text:
                # Generate text from other fields
                text = (
                    f"Group {rate.get('vehicle_group', 'Unknown')} vehicle hire "
                    f"in {rate.get('region', 'UK')}, "
                    f"£{rate.get('daily_rate', 0)}/day, "
                    f"{rate.get('year', 2024)}"
                )
            
            documents.append(text)
            
            # Metadata for filtering
            metadata = {
                "vehicle_group": rate.get("vehicle_group"),
                "region": rate.get("region"),
                "daily_rate": rate.get("daily_rate"),
                "year": rate.get("year"),
                "company": rate.get("company")
            }
            # Remove None values
            metadata = {k: v for k, v in metadata.items() if v is not None}
            metadatas.append(metadata)
        
        # Add to vector store
        ids = store.add_documents(documents=documents, metadatas=metadatas)
        
        return JSONResponse(content={
            "success": True,
            "added": len(ids),
            "total_count": store.count
        })
        
    except Exception as e:
        logger.error(f"Failed to add rates: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add rates: {str(e)}"
        )


@app.get("/rates/search", tags=["Rates"])
async def search_rates(
    query: str = Query(..., description="Search query"),
    vehicle_group: Optional[str] = Query(None, description="Filter by vehicle group"),
    top_k: int = Query(10, description="Number of results")
):
    """
    Search for rates by similarity.
    
    Returns rates similar to the query, optionally filtered by vehicle group.
    """
    store = get_rates_store()
    
    filters = {}
    if vehicle_group:
        filters["vehicle_group"] = vehicle_group.upper()
    
    results = store.search(
        query=query,
        top_k=top_k,
        where=filters if filters else None
    )
    
    return JSONResponse(content={
        "query": query,
        "total_found": results.total_found,
        "results": [
            {
                "id": r.id,
                "text": r.document,
                "metadata": r.metadata,
                "similarity": round(r.similarity, 3)
            }
            for r in results.results
        ]
    })


@app.get("/rates/summary/{vehicle_group}", response_model=MarketSummaryResponse, tags=["Rates"])
async def get_market_summary(
    vehicle_group: str,
    region: Optional[str] = Query(None, description="Optional region filter")
):
    """
    Get market rate summary for a vehicle group.
    
    Returns statistics: min, max, mean, median rates.
    """
    from rate_matcher import get_rate_matcher
    
    matcher = get_rate_matcher()
    summary = matcher.get_market_summary(
        vehicle_group=vehicle_group.upper(),
        region=region
    )
    
    return MarketSummaryResponse(**summary)


@app.get("/rates/stats", tags=["Rates"])
async def get_rate_stats():
    """
    Get overall statistics about the rate database.
    """
    store = get_rates_store()
    
    return JSONResponse(content={
        "total_rates": store.count,
        "collection_name": settings.rates_collection_name
    })


# -----------------------------------------------------------------------------
# System Endpoints
# -----------------------------------------------------------------------------
@app.get("/models", tags=["System"])
async def list_models():
    """
    List active Bedrock models.
    """
    llm = get_llm_client()
    models = llm.list_models()
    
    return JSONResponse(content={
        "current_model": settings.bedrock_llm_model_id,
        "available_models": models,
        "bedrock_available": llm.is_available()
    })


@app.get("/config", tags=["System"])
async def get_config():
    """
    Get current configuration (non-sensitive values).
    """
    return JSONResponse(content={
        "bedrock_llm_model": settings.bedrock_llm_model_id,
        "bedrock_embedding_model": settings.bedrock_embedding_model_id,
        "rag_top_k": settings.rag_top_k,
        "fair_threshold": settings.fair_threshold,
        "flagged_threshold": settings.flagged_threshold,
        "debug": settings.debug
    })


# -----------------------------------------------------------------------------
# Error Handlers
# -----------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Handle any unhandled exceptions.
    """
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An error occurred"
        }
    )


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )