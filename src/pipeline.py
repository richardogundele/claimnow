"""
pipeline.py - Full Claim Analysis Pipeline

WHY THIS FILE EXISTS:
- Orchestrate all components into one workflow
- Single entry point for analyzing a claim
- Handle errors gracefully at each stage
- Track timing and performance

THE PIPELINE STAGES:
1. PARSE: Extract text from PDF (document_parser.py)
2. EXTRACT: Get structured fields using LLM (extractor.py)
3. MATCH: Find comparable market rates (rate_matcher.py)
4. SCORE: Classify the claim (scorer.py)
5. EXPLAIN: Generate human explanation (explainer.py)

DESIGN PATTERN:
This uses the "Pipeline" pattern - each stage transforms data
and passes it to the next. Errors at any stage are captured
and the pipeline continues with partial results.

USAGE:
    pipeline = ClaimsPipeline()
    result = pipeline.analyze("court_pack.pdf")
    
    print(result.verdict)
    print(result.explanation)
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Import from sibling modules using package-relative imports
# 'src.' prefix required when running from project root
from src.document_parser import DocumentParser, DocumentContent, parse_document
from src.extractor import ClaimExtractor, ExtractedClaim, extract_claim
from src.rate_matcher import RateMatcher, RateMatchResult, match_rate, RateComparison
from src.scorer import ClaimScorer, ScoringResult, Verdict, score_claim
from src.explainer import ClaimExplainer, Explanation, explain_score
from src.config import settings

# Set up logging
logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """
    Stages of the analysis pipeline.
    
    Used for tracking progress and identifying where errors occurred.
    """
    PARSE = "parse"
    EXTRACT = "extract"
    MATCH = "match"
    SCORE = "score"
    EXPLAIN = "explain"
    COMPLETE = "complete"


@dataclass
class StageResult:
    """
    Result of a single pipeline stage.
    
    Tracks success, timing, and any errors.
    """
    stage: PipelineStage
    success: bool = True
    duration_ms: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage": self.stage.value,
            "success": self.success,
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error
        }


@dataclass
class AnalysisResult:
    """
    Complete result of analyzing a claim.
    
    Contains all outputs from each pipeline stage.
    """
    # Identification
    claim_id: str = ""
    filename: str = ""
    analyzed_at: str = ""
    
    # Final verdict
    verdict: Verdict = Verdict.INSUFFICIENT_DATA
    confidence: float = 0.0
    
    # Stage outputs
    document: Optional[DocumentContent] = None
    extracted_claim: Optional[ExtractedClaim] = None
    rate_match: Optional[RateMatchResult] = None
    scoring: Optional[ScoringResult] = None
    explanation: Optional[Explanation] = None
    
    # Pipeline metadata
    stages: Dict[str, StageResult] = field(default_factory=dict)
    total_duration_ms: float = 0.0
    completed_stages: int = 0
    failed_stage: Optional[str] = None
    
    # Summary for display
    summary: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "claim_id": self.claim_id,
            "filename": self.filename,
            "analyzed_at": self.analyzed_at,
            "verdict": self.verdict.value,
            "confidence": round(self.confidence, 3),
            "summary": self.summary,
            "extracted_claim": self.extracted_claim.to_dict() if self.extracted_claim else None,
            "rate_match": self.rate_match.to_dict() if self.rate_match else None,
            "scoring": self.scoring.to_dict() if self.scoring else None,
            "explanation": self.explanation.to_dict() if self.explanation else None,
            "pipeline": {
                "total_duration_ms": round(self.total_duration_ms, 2),
                "completed_stages": self.completed_stages,
                "failed_stage": self.failed_stage,
                "stages": {k: v.to_dict() for k, v in self.stages.items()}
            }
        }
    
    @property
    def is_complete(self) -> bool:
        """Check if all stages completed successfully."""
        return self.failed_stage is None and self.completed_stages == 5
    
    @property
    def verdict_color(self) -> str:
        """Get color code for verdict (for UI)."""
        colors = {
            Verdict.FAIR: "green",
            Verdict.POTENTIALLY_INFLATED: "yellow",
            Verdict.FLAGGED: "red",
            Verdict.INSUFFICIENT_DATA: "gray"
        }
        return colors.get(self.verdict, "gray")


class ClaimsPipeline:
    """
    Full claim analysis pipeline.
    
    USAGE:
        pipeline = ClaimsPipeline()
        
        # Analyze a PDF file
        result = pipeline.analyze("court_pack.pdf")
        
        # Or analyze from bytes (uploaded file)
        result = pipeline.analyze_bytes(pdf_bytes, "uploaded.pdf")
        
        # Check result
        print(f"Verdict: {result.verdict.value}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Summary: {result.summary}")
    
    CONFIGURATION:
        pipeline = ClaimsPipeline(
            skip_explanation=True,  # Skip SHAP for speed
            verbose=True            # Log each stage
        )
    """
    
    def __init__(
        self,
        parser: Optional[DocumentParser] = None,
        extractor: Optional[ClaimExtractor] = None,
        matcher: Optional[RateMatcher] = None,
        scorer: Optional[ClaimScorer] = None,
        explainer: Optional[ClaimExplainer] = None,
        skip_explanation: bool = False,
        verbose: bool = True
    ):
        """
        Initialize the pipeline with components.
        
        Args:
            parser: Document parser (default: create new)
            extractor: Claim extractor (default: create new)
            matcher: Rate matcher (default: create new)
            scorer: Claim scorer (default: create new)
            explainer: Claim explainer (default: create new)
            skip_explanation: Skip SHAP explanation for faster processing
            verbose: Log progress at each stage
        """
        self.parser = parser or DocumentParser()
        self.extractor = extractor or ClaimExtractor()
        self.matcher = matcher or RateMatcher()
        self.scorer = scorer or ClaimScorer()
        self.explainer = explainer or ClaimExplainer() if not skip_explanation else None
        
        self.skip_explanation = skip_explanation
        self.verbose = verbose
        
        logger.info("ClaimsPipeline initialized")
    
    def analyze(
        self,
        file_path: Union[str, Path],
        claim_id: Optional[str] = None
    ) -> AnalysisResult:
        """
        Analyze a claim document from file path.
        
        This is the MAIN METHOD for file-based analysis.
        
        Args:
            file_path: Path to the PDF file
            claim_id: Optional ID for the claim (auto-generated if not provided)
            
        Returns:
            AnalysisResult with all outputs
        """
        file_path = Path(file_path)
        
        # Generate claim ID if not provided
        if claim_id is None:
            claim_id = self._generate_claim_id(file_path.name)
        
        # Initialize result
        result = AnalysisResult(
            claim_id=claim_id,
            filename=file_path.name,
            analyzed_at=datetime.now().isoformat()
        )
        
        pipeline_start = time.time()
        
        # Stage 1: Parse document
        result = self._run_stage(
            result,
            PipelineStage.PARSE,
            lambda: self._parse_document(file_path)
        )
        
        if result.document is None:
            return self._finalize_result(result, pipeline_start)
        
        # Continue with common processing
        return self._process_document(result, pipeline_start)
    
    def analyze_bytes(
        self,
        pdf_bytes: bytes,
        filename: str = "uploaded.pdf",
        claim_id: Optional[str] = None
    ) -> AnalysisResult:
        """
        Analyze a claim document from bytes.
        
        Used for uploaded files via the API.
        
        Args:
            pdf_bytes: Raw PDF file bytes
            filename: Name to assign
            claim_id: Optional claim ID
            
        Returns:
            AnalysisResult with all outputs
        """
        if claim_id is None:
            claim_id = self._generate_claim_id(filename)
        
        result = AnalysisResult(
            claim_id=claim_id,
            filename=filename,
            analyzed_at=datetime.now().isoformat()
        )
        
        pipeline_start = time.time()
        
        # Stage 1: Parse document from bytes
        result = self._run_stage(
            result,
            PipelineStage.PARSE,
            lambda: self.parser.parse_bytes(pdf_bytes, filename)
        )
        
        if result.document is None:
            return self._finalize_result(result, pipeline_start)
        
        # Continue with common processing
        return self._process_document(result, pipeline_start)
    
    def analyze_text(
        self,
        document_text: str,
        filename: str = "text_input",
        claim_id: Optional[str] = None
    ) -> AnalysisResult:
        """
        Analyze a claim from raw text (skips parsing).
        
        Useful for testing or when text is already extracted.
        
        Args:
            document_text: Raw document text
            filename: Name to assign
            claim_id: Optional claim ID
            
        Returns:
            AnalysisResult with all outputs
        """
        if claim_id is None:
            claim_id = self._generate_claim_id(filename)
        
        result = AnalysisResult(
            claim_id=claim_id,
            filename=filename,
            analyzed_at=datetime.now().isoformat()
        )
        
        # Create mock document content
        result.document = DocumentContent(
            filename=filename,
            total_pages=1,
            pages=[],
            full_text=document_text
        )
        
        # Mark parse stage as complete
        result.stages[PipelineStage.PARSE.value] = StageResult(
            stage=PipelineStage.PARSE,
            success=True,
            duration_ms=0.0
        )
        result.completed_stages = 1
        
        pipeline_start = time.time()
        
        # Continue with extraction and beyond
        return self._process_document(result, pipeline_start, skip_parse=True)
    
    def _process_document(
        self,
        result: AnalysisResult,
        pipeline_start: float,
        skip_parse: bool = False
    ) -> AnalysisResult:
        """
        Process a parsed document through remaining stages.
        """
        # Stage 2: Extract fields
        result = self._run_stage(
            result,
            PipelineStage.EXTRACT,
            lambda: self.extractor.extract(result.document.full_text)
        )
        
        if result.extracted_claim is None:
            return self._finalize_result(result, pipeline_start)
        
        # Stage 3: Match rates
        result = self._run_stage(
            result,
            PipelineStage.MATCH,
            lambda: self.matcher.match_claim(result.extracted_claim)
        )
        
        if result.rate_match is None:
            return self._finalize_result(result, pipeline_start)
        
        # Stage 4: Score claim
        result = self._run_stage(
            result,
            PipelineStage.SCORE,
            lambda: self.scorer.score(result.extracted_claim, result.rate_match, claim_id=result.claim_id)
        )
        
        if result.scoring is None:
            return self._finalize_result(result, pipeline_start)
        
        # Update verdict from scoring
        result.verdict = result.scoring.verdict
        result.confidence = result.scoring.confidence
        
        # Stage 5: Explain (optional)
        if not self.skip_explanation and self.explainer:
            result = self._run_stage(
                result,
                PipelineStage.EXPLAIN,
                lambda: self.explainer.explain(
                    result.scoring,
                    result.extracted_claim,
                    result.rate_match
                )
            )
        else:
            # Skip explanation stage
            result.stages[PipelineStage.EXPLAIN.value] = StageResult(
                stage=PipelineStage.EXPLAIN,
                success=True,
                duration_ms=0.0
            )
            result.completed_stages += 1
        
        return self._finalize_result(result, pipeline_start)
    
    def _run_stage(
        self,
        result: AnalysisResult,
        stage: PipelineStage,
        func: callable
    ) -> AnalysisResult:
        """
        Run a single pipeline stage with timing and error handling.
        """
        stage_start = time.time()
        
        if self.verbose:
            logger.info(f"Starting stage: {stage.value}")
        
        try:
            output = func()
            duration_ms = (time.time() - stage_start) * 1000
            
            # Store output in appropriate field
            if stage == PipelineStage.PARSE:
                result.document = output
            elif stage == PipelineStage.EXTRACT:
                result.extracted_claim = output
            elif stage == PipelineStage.MATCH:
                result.rate_match = output
            elif stage == PipelineStage.SCORE:
                result.scoring = output
            elif stage == PipelineStage.EXPLAIN:
                result.explanation = output
            
            result.stages[stage.value] = StageResult(
                stage=stage,
                success=True,
                duration_ms=duration_ms
            )
            result.completed_stages += 1
            
            if self.verbose:
                logger.info(f"Completed stage: {stage.value} ({duration_ms:.0f}ms)")
            
        except Exception as e:
            duration_ms = (time.time() - stage_start) * 1000
            
            logger.error(f"Stage {stage.value} failed: {e}")
            
            result.stages[stage.value] = StageResult(
                stage=stage,
                success=False,
                duration_ms=duration_ms,
                error=str(e)
            )
            result.failed_stage = stage.value
        
        return result
    
    def _parse_document(self, file_path: Path) -> DocumentContent:
        """Parse a document file."""
        return self.parser.parse(file_path)
    
    def _finalize_result(
        self,
        result: AnalysisResult,
        pipeline_start: float
    ) -> AnalysisResult:
        """
        Finalize the result with summary and totals.
        """
        result.total_duration_ms = (time.time() - pipeline_start) * 1000
        
        # Generate summary
        result.summary = self._generate_summary(result)
        
        if self.verbose:
            status = "COMPLETE" if result.is_complete else f"FAILED at {result.failed_stage}"
            logger.info(
                f"Pipeline {status} in {result.total_duration_ms:.0f}ms - "
                f"Verdict: {result.verdict.value}"
            )
        
        return result
    
    def _generate_summary(self, result: AnalysisResult) -> str:
        """
        Generate a human-readable summary of the analysis.
        """
        parts = []
        
        # Verdict
        if result.verdict == Verdict.FAIR:
            parts.append("This claim appears to be FAIR and within market norms.")
        elif result.verdict == Verdict.POTENTIALLY_INFLATED:
            parts.append("This claim is POTENTIALLY INFLATED above market rates.")
        elif result.verdict == Verdict.FLAGGED:
            parts.append("This claim has been FLAGGED for significant concerns.")
        else:
            parts.append("Unable to make a reliable assessment due to insufficient data.")
        
        # Confidence
        if result.confidence > 0:
            parts.append(f"Confidence: {result.confidence:.0%}.")
        
        # Rate comparison
        if result.rate_match and result.rate_match.statistics.count > 0:
            rm = result.rate_match
            parts.append(
                f"Claimed rate £{rm.claimed_rate:.2f}/day "
                f"vs market average £{rm.statistics.mean_rate:.2f}/day "
                f"({rm.deviation_percent:+.1f}%)."
            )
        
        # Key extraction info
        if result.extracted_claim:
            ec = result.extracted_claim
            if ec.vehicle.group:
                parts.append(f"Vehicle Group {ec.vehicle.group}.")
            if ec.hire_period.duration_days:
                parts.append(f"Hire duration: {ec.hire_period.duration_days} days.")
        
        # Processing time
        parts.append(f"Analyzed in {result.total_duration_ms:.0f}ms.")
        
        return " ".join(parts)
    
    def _generate_claim_id(self, filename: str) -> str:
        """
        Generate a unique claim ID.
        """
        import hashlib
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_hash = hashlib.md5(filename.encode()).hexdigest()[:6]
        return f"CLM-{timestamp}-{file_hash}"


# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------
_default_pipeline: Optional[ClaimsPipeline] = None


def get_pipeline() -> ClaimsPipeline:
    """Get the default pipeline (singleton)."""
    global _default_pipeline
    
    if _default_pipeline is None:
        _default_pipeline = ClaimsPipeline()
    
    return _default_pipeline


def analyze_claim(file_path: Union[str, Path]) -> AnalysisResult:
    """
    Quick function to analyze a claim file.
    
    USAGE:
        from pipeline import analyze_claim
        
        result = analyze_claim("court_pack.pdf")
        print(result.verdict.value)
    """
    pipeline = get_pipeline()
    return pipeline.analyze(file_path)


def analyze_claim_text(text: str) -> AnalysisResult:
    """
    Quick function to analyze claim from text.
    
    USAGE:
        from pipeline import analyze_claim_text
        
        text = "Invoice... Daily rate £65... Vehicle: BMW 3 Series..."
        result = analyze_claim_text(text)
    """
    pipeline = get_pipeline()
    return pipeline.analyze_text(text)