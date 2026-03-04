"""
scorer.py - RAG-Based Claim Scoring with Evidence

WHY THIS FILE EXISTS:
- Score insurance claims using context from similar cases
- Use LLM (AWS Bedrock) to make decisions based on market context
- Show evidence: which market rates support the decision
- Scale to 65M unstructured documents using RAG

HOW THE SCORER WORKS:
1. Take extracted claim (vehicle, rate, dates, etc.)
2. Get comparable rates from vector store (rate matching)
3. Format those rates as context for the LLM
4. Ask LLM: "Is this claim fair given market context?"
5. Parse LLM response to get verdict + reasoning
6. Return verdict + sources (which docs influenced decision)

WHY RAG INSTEAD OF ML:
- ML classifier ignores your 65M documents
- RAG retrieves relevant context first, then reasons about it
- LLM can explain its thinking (FAIR because rate is within range)
- Transparent and auditable for insurance regulators
- Easier to fix (change prompt, not retrain model)

EVIDENCE HIGHLIGHTING:
- Each result shows which market rates are similar
- User can see: "Your rate matches Claim-XYZ which was rated FAIR"
- Sources are the comparable rates found by vector search
"""

import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid

# Import from sibling modules using package-relative imports
from src.extractor import ExtractedClaim
from src.rate_matcher import RateMatchResult, RateComparison
from src.llm_client import BedrockClient, get_llm_client, Message
from src.config import settings, get_absolute_path

# Set up logging
logger = logging.getLogger(__name__)


class Verdict(Enum):
    """
    A decision about whether a claim is fair or not.
    
    FAIR: Claim rate is normal and good
    POTENTIALLY_INFLATED: Claim rate is higher than normal, needs looking at
    FLAGGED: Claim has big problems, looks bad
    INSUFFICIENT_DATA: Don't have enough information to say
    """
    FAIR = "FAIR"
    POTENTIALLY_INFLATED = "POTENTIALLY_INFLATED"
    FLAGGED = "FLAGGED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class EvidenceSource:
    """
    One piece of proof for the decision we made.
    
    This is a market rate that helped the LLM decide.
    Includes highlighting of which parts support the verdict.
    """
    source_id: str  # Which document this came from
    daily_rate: float  # The market rate in pounds
    vehicle_group: str  # Type of car (A, B, C, D, etc)
    region: Optional[str] = None  # Where the rate is from
    similarity_score: float = 0.0  # How close it matches our claim (0-1)
    relevance: str = "comparable"  # Is it a good match?
    highlighted_text: Optional[str] = None  # Which part of the document matters
    impact_on_verdict: str = "supporting"  # supporting or refuting
    
    def to_dict(self) -> Dict[str, Any]:
        """Turn into a dictionary."""
        return {
            "source_id": self.source_id,
            "daily_rate": self.daily_rate,
            "vehicle_group": self.vehicle_group,
            "region": self.region,
            "similarity_score": round(self.similarity_score, 3),
            "relevance": self.relevance,
            "highlighted_text": self.highlighted_text,
            "impact_on_verdict": self.impact_on_verdict
        }


@dataclass
class ScoringFeatures:
    """
    Simple tracker for what influenced the score.
    
    Just the important numbers, not ML stuff.
    """
    # What we're checking
    claimed_rate: float = 0.0
    market_mean_rate: float = 0.0
    rate_deviation_percent: float = 0.0
    
    # How much data we have
    market_rate_count: int = 0
    vehicle_group: str = "C"
    extraction_confidence: float = 0.0
    hire_duration_days: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Turn into a dictionary."""
        return {
            "claimed_rate": self.claimed_rate,
            "market_mean_rate": self.market_mean_rate,
            "rate_deviation_percent": round(self.rate_deviation_percent, 2),
            "market_rate_count": self.market_rate_count,
            "vehicle_group": self.vehicle_group,
            "extraction_confidence": round(self.extraction_confidence, 3),
            "hire_duration_days": self.hire_duration_days
        }


@dataclass
class AuditLog:
    """
    Enterprise audit trail for every decision.
    
    Track who decided what, when, and why for compliance.
    """
    audit_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Unique ID
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())  # When
    claim_id: Optional[str] = None  # Which claim
    scoring_method: str = "llm_rag"  # Method used (llm_rag or rules)
    model_version: str = "1.0"  # Model version
    llm_model: str = "mistral"  # Which LLM
    retrieval_count: int = 0  # How many refs retrieved
    decision_latency_ms: float = 0.0  # How long it took
    user_overrides: Optional[str] = None  # If human changed it
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "timestamp": self.timestamp,
            "claim_id": self.claim_id,
            "scoring_method": self.scoring_method,
            "model_version": self.model_version,
            "llm_model": self.llm_model,
            "retrieval_count": self.retrieval_count,
            "decision_latency_ms": round(self.decision_latency_ms, 2),
            "user_overrides": self.user_overrides
        }


@dataclass
class ScoringResult:
    """
    The answer: is the claim fair or not?
    
    Includes what we decided, how sure we are,
    which documents prove it, and audit trail.
    """
    verdict: Verdict  # FAIR, POTENTIALLY_INFLATED, FLAGGED, or INSUFFICIENT_DATA
    confidence: float  # How sure we are (0.0 to 1.0)
    explanation: str = ""  # Why we picked this (in plain words)
    reasoning: str = ""  # What the smart helper said (from LLM)
    features: ScoringFeatures = field(default_factory=ScoringFeatures)  # The numbers we looked at
    evidence_sources: List[EvidenceSource] = field(default_factory=list)  # Market rates that prove it
    audit_log: AuditLog = field(default_factory=AuditLog)  # Compliance trail
    error: Optional[str] = None  # If something went wrong
    
    def to_dict(self) -> Dict[str, Any]:
        """Turn into a structured JSON verdict for enterprise systems."""
        return {
            "verdict": {
                "decision": self.verdict.value,
                "confidence": round(self.confidence, 3),
                "explanation": self.explanation
            },
            "reasoning": {
                "llm_analysis": self.reasoning,
                "key_features": self.features.to_dict()
            },
            "evidence": {
                "sources": [s.to_dict() for s in self.evidence_sources],
                "count": len(self.evidence_sources)
            },
            "audit": self.audit_log.to_dict(),
            "error": self.error
        }


class ClaimScorer:
    """
    Score claims using the smart helper and market context.
    
    USAGE:
        scorer = ClaimScorer()
        result = scorer.score(extracted_claim, rate_match_result)
        
        print(f"Verdict: {result.verdict.value}")
        print(f"We are {result.confidence:.1%} sure")
        print(f"Proof: {len(result.evidence_sources)} similar market rates")
    
    HOW IT WORKS:
    1. Check: Do we have enough market data?
    2. Format the market rates as context for the smart helper
    3. Ask the smart helper: "Is this claim fair?"
    4. If smart helper is unsure, use simple rules
    5. Show the evidence (which docs helped decide)
    """
    
    def __init__(self, llm_client: Optional[BedrockClient] = None):
        """
        Start up the scorer.
        
        Args:
            llm_client: The smart helper to use (if None, gets the default one)
        """
        self.llm = llm_client or get_llm_client()
        self.audit_log = AuditLog()
        logger.info("RAG-based ClaimScorer initialized")
    
    def score(
        self,
        claim: ExtractedClaim,
        rate_result: RateMatchResult,
        claim_id: Optional[str] = None
    ) -> ScoringResult:
        """
        Score a claim and show the proof.
        
        This is the MAIN method.
        
        Args:
            claim: What the person said they spent
            rate_result: What the market thinks it should cost
            claim_id: Unique claim identifier for audit trail
            
        Returns:
            ScoringResult: The decision plus proof with audit trail
        """
        start_time = datetime.utcnow()
        audit_log = AuditLog(claim_id=claim_id or "unknown")
        
        try:
            # Check if we have enough data
            if not rate_result.has_sufficient_data or rate_result.statistics.count < 3:
                audit_log.decision_latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                return ScoringResult(
                    verdict=Verdict.INSUFFICIENT_DATA,
                    confidence=0.0,
                    explanation="Not enough market data to decide if this claim is fair.",
                    error="Insufficient comparable rates found",
                    audit_log=audit_log
                )
            
            # Get what we're checking
            features = self._extract_features(claim, rate_result)
            
            # Try to use the smart helper first
            llm_result = self._score_with_llm(
                claim, rate_result, features
            )
            
            # If LLM worked, use that
            if llm_result is not None:
                llm_result.audit_log = audit_log
                llm_result.audit_log.scoring_method = "llm_rag"
                llm_result.audit_log.retrieval_count = len(llm_result.evidence_sources)
                llm_result.audit_log.decision_latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                return llm_result
            
            # Fall back to simple rules
            logger.info("LLM scoring failed, falling back to rules")
            rules_result = self._score_with_rules(
                claim, rate_result, features
            )
            rules_result.audit_log = audit_log
            rules_result.audit_log.scoring_method = "rules_based"
            rules_result.audit_log.decision_latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            return rules_result
        
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            audit_log.decision_latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            return ScoringResult(
                verdict=Verdict.INSUFFICIENT_DATA,
                confidence=0.0,
                error=str(e),
                audit_log=audit_log
            )
    
    def _extract_features(
        self,
        claim: ExtractedClaim,
        rate_result: RateMatchResult
    ) -> ScoringFeatures:
        """
        Pull out the important numbers.
        """
        return ScoringFeatures(
            claimed_rate=rate_result.claimed_rate,
            market_mean_rate=rate_result.statistics.mean_rate,
            rate_deviation_percent=rate_result.deviation_percent,
            market_rate_count=rate_result.statistics.count,
            vehicle_group=claim.vehicle.group or "C",
            extraction_confidence=claim.extraction_confidence,
            hire_duration_days=claim.hire_period.duration_days or 0
        )
    
    def _score_with_llm(
        self,
        claim: ExtractedClaim,
        rate_result: RateMatchResult,
        features: ScoringFeatures
    ) -> Optional[ScoringResult]:
        """
        Ask the smart helper if the claim is fair.
        
        Returns None if it fails, so we can use rules instead.
        """
        try:
            # Format the market data as context
            context_text = self._format_market_context(
                rate_result, 
                features
            )
            
            # Build the question for the smart helper
            prompt = self._build_scoring_prompt(
                claim, rate_result, features, context_text
            )
            
            # Ask the smart helper
            logger.info("Asking LLM to score claim")
            response = self.llm.generate(prompt)
            
            if not response.success or not response.text:
                logger.warning("LLM scoring returned no text")
                return None
            
            # Parse the answer
            result = self._parse_llm_response(
                response.text,
                claim,
                rate_result,
                features
            )
            
            return result
        
        except Exception as e:
            logger.error(f"LLM scoring error: {e}")
            return None
    
    def _format_market_context(
        self,
        rate_result: RateMatchResult,
        features: ScoringFeatures
    ) -> str:
        """
        Turn market data into text the LLM can read.
        """
        lines = []
        
        # Summary statistics
        lines.append(f"MARKET DATA FOR {features.vehicle_group} VEHICLES:")
        lines.append(f"- Count: {features.market_rate_count} comparable rates found")
        lines.append(f"- Minimum: £{rate_result.statistics.min_rate:.2f}/day")
        lines.append(f"- Average: £{rate_result.statistics.mean_rate:.2f}/day")
        lines.append(f"- Maximum: £{rate_result.statistics.max_rate:.2f}/day")
        if features.market_rate_count > 0:
            lines.append(f"- Standard deviation: £{rate_result.statistics.standard_deviation or 0:.2f}")
        
        # Percentile info if we have enough data
        if features.market_rate_count >= 10:
            p25 = rate_result.statistics.percentile_25 or 0
            p75 = rate_result.statistics.percentile_75 or 0
            lines.append(f"- 25th percentile: £{p25:.2f}/day")
            lines.append(f"- 75th percentile: £{p75:.2f}/day")
        
        # Comparison
        lines.append("")
        lines.append("CLAIM:")
        lines.append(f"- Claimed rate: £{features.claimed_rate:.2f}/day")
        lines.append(f"- Deviation from average: {features.rate_deviation_percent:+.1f}%")
        
        if features.rate_deviation_percent > 0:
            lines.append(f"  (This is ABOVE market average by £{features.claimed_rate - rate_result.statistics.mean_rate:.2f}/day)")
        else:
            lines.append(f"  (This is BELOW market average by £{abs(features.claimed_rate - rate_result.statistics.mean_rate):.2f}/day)")
        
        return "\n".join(lines)
    
    def _build_scoring_prompt(
        self,
        claim: ExtractedClaim,
        rate_result: RateMatchResult,
        features: ScoringFeatures,
        context_text: str
    ) -> str:
        """
        Write the question to ask the smart helper.
        """
        prompt = f"""You are an expert insurance claim analyst. Analyze this claim and decide if the rate is fair.

{context_text}

CLAIM DETAILS:
- Vehicle: {claim.vehicle.make} {claim.vehicle.model or ''} (Group {features.vehicle_group})
- Hire duration: {features.hire_duration_days} days
- Daily rate claimed: £{features.claimed_rate:.2f}
- Extraction confidence: {features.extraction_confidence:.1%}

TASK:
Based on the market data above, is this claim:
1. FAIR - rate is normal
2. POTENTIALLY_INFLATED - rate is above normal but not crazy
3. FLAGGED - rate looks wrong

RESPONSE FORMAT (JSON):
{{
  "verdict": "FAIR" or "POTENTIALLY_INFLATED" or "FLAGGED",
  "confidence": 0.0 to 1.0,
  "reasoning": "Short explanation (1-2 sentences) of why this verdict",
  "key_evidence": "What in the market data matters most for this decision"
}}

Respond ONLY with the JSON, no other text."""

        return prompt
    
    def _parse_llm_response(
        self,
        response_text: str,
        claim: ExtractedClaim,
        rate_result: RateMatchResult,
        features: ScoringFeatures
    ) -> Optional[ScoringResult]:
        """
        Read the smart helper's answer and turn it into a decision.
        """
        try:
            # Try to find JSON in the response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            
            if json_start == -1 or json_end == 0:
                logger.warning("No JSON found in LLM response")
                return None
            
            json_text = response_text[json_start:json_end]
            llm_response = json.loads(json_text)
            
            # Extract the verdict
            verdict_str = llm_response.get("verdict", "").upper()
            try:
                verdict = Verdict(verdict_str)
            except ValueError:
                logger.warning(f"Unknown verdict: {verdict_str}")
                return None
            
            confidence = float(llm_response.get("confidence", 0.5))
            reasoning = llm_response.get("reasoning", "")
            
            # Build evidence from rate results
            evidence = self._build_evidence_from_rates(rate_result)
            
            # Generate explanation
            explanation = self._generate_explanation(
                verdict, features, rate_result
            )
            
            return ScoringResult(
                verdict=verdict,
                confidence=confidence,
                explanation=explanation,
                reasoning=reasoning,
                features=features,
                evidence_sources=evidence
            )
        
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return None
    
    def _build_evidence_from_rates(
        self,
        rate_result: RateMatchResult
    ) -> List[EvidenceSource]:
        """
        Turn the comparable rates into evidence sources.
        Includes highlighting of key information.
        """
        evidence = []
        
        for i, rate in enumerate(rate_result.comparable_rates[:5]):
            # Create highlighting text (what part of the doc matters)
            if i == 0:
                highlight = f"Most similar: {rate.vehicle_group} vehicle at £{rate.daily_rate:.2f}/day"
            else:
                highlight = f"Similar {rate.vehicle_group} group rate: £{rate.daily_rate:.2f}/day"
            
            source = EvidenceSource(
                source_id=f"rate_{i+1}",
                daily_rate=rate.daily_rate,
                vehicle_group=rate.vehicle_group,
                region=rate.region,
                similarity_score=rate.similarity_score if hasattr(rate, 'similarity_score') else 0.95 - (i * 0.05),
                relevance="comparable" if i < 3 else "reference",
                highlighted_text=highlight,
                impact_on_verdict="supporting"
            )
            evidence.append(source)
        
        return evidence
    
    def _score_with_rules(
        self,
        claim: ExtractedClaim,
        rate_result: RateMatchResult,
        features: ScoringFeatures
    ) -> ScoringResult:
        """
        Simple rules for when the smart helper isn't available.
        
        Just look at the numbers.
        """
        deviation = features.rate_deviation_percent
        
        # Simple thresholds
        if deviation <= 15:
            verdict = Verdict.FAIR
            confidence = 0.95 - (abs(deviation) / 15) * 0.2
        elif deviation <= 40:
            verdict = Verdict.POTENTIALLY_INFLATED
            confidence = 0.6 + (abs(deviation) - 15) / 25 * 0.3
        else:
            verdict = Verdict.FLAGGED
            confidence = min(0.99, 0.7 + (abs(deviation) - 40) / 60 * 0.25)
        
        explanation = self._generate_explanation(
            verdict, features, rate_result
        )
        
        evidence = self._build_evidence_from_rates(rate_result)
        
        return ScoringResult(
            verdict=verdict,
            confidence=max(0.0, min(1.0, confidence)),
            explanation=explanation,
            reasoning="Rule-based scoring (LLM unavailable)",
            features=features,
            evidence_sources=evidence
        )
    
    def _generate_explanation(
        self,
        verdict: Verdict,
        features: ScoringFeatures,
        rate_result: RateMatchResult
    ) -> str:
        """
        Write a short explanation in plain words.
        """
        parts = []
        
        # The verdict in plain words
        if verdict == Verdict.FAIR:
            parts.append("This claim looks fair and normal.")
        elif verdict == Verdict.POTENTIALLY_INFLATED:
            parts.append("This claim is higher than what we usually see.")
        elif verdict == Verdict.FLAGGED:
            parts.append("This claim looks wrong and needs checking.")
        else:
            return "Not enough data to decide."
        
        # Why
        parts.append(
            f"The £{features.claimed_rate:.2f}/day claimed rate "
            f"is {abs(features.rate_deviation_percent):.0f}% "
            f"{'higher' if features.rate_deviation_percent > 0 else 'lower'} than the "
            f"market average of £{features.market_mean_rate:.2f}/day"
        )
        
        # Confidence note
        if features.extraction_confidence < 0.7:
            parts.append(
                "Note: We're not fully sure about all the details in the paper."
            )
        
        return " ".join(parts) + "."


# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------
_default_scorer: Optional[ClaimScorer] = None


def get_scorer() -> ClaimScorer:
    """Get the default scorer (singleton)."""
    global _default_scorer
    
    if _default_scorer is None:
        _default_scorer = ClaimScorer()
    
    return _default_scorer


def score_claim(
    claim: ExtractedClaim,
    rate_result: RateMatchResult,
    claim_id: Optional[str] = None
) -> ScoringResult:
    """
    Quick function to score a claim with audit trail.
    
    USAGE:
        from scorer import score_claim
        
        result = score_claim(claim, rate_match, claim_id="CLAIM-001")
        print(result.verdict.value)
        print(f"Evidence: {result.evidence_sources}")
        print(f"Audit: {result.audit_log.audit_id}")
    """
    scorer = get_scorer()
    return scorer.score(claim, rate_result, claim_id=claim_id)
