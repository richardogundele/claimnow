"""
explainer.py - Model Explainability with SHAP

WHY THIS FILE EXISTS:
- Provide transparent explanations for every scoring decision
- Use SHAP (SHapley Additive exPlanations) for feature importance
- Generate human-readable summaries
- Meet regulatory requirements for AI explainability

WHAT IS SHAP:
- SHAP values show how each feature contributes to a prediction
- Based on game theory (Shapley values)
- Positive SHAP = pushes prediction higher (toward FLAGGED)
- Negative SHAP = pushes prediction lower (toward FAIR)
- Sum of SHAP values = difference from average prediction

WHY EXPLAINABILITY MATTERS:
- Insurance decisions affect real people
- Regulators require explanations (GDPR Article 22)
- Builds trust with users
- Helps identify model bias or errors

EXAMPLE OUTPUT:
"The claim was flagged because:
 - Rate deviation (+32%) contributed +0.25 to the score
 - Low extraction confidence contributed +0.12
 - Long hire duration (45 days) contributed +0.08"
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

# Import from sibling modules using package-relative imports
from src.scorer import ClaimScorer, ScoringFeatures, ScoringResult, Verdict, get_scorer
from src.extractor import ExtractedClaim
from src.rate_matcher import RateMatchResult

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class FeatureContribution:
    """
    Contribution of a single feature to the prediction.
    
    Attributes:
        feature_name: Name of the feature
        feature_value: The actual value of the feature
        shap_value: SHAP contribution (positive = toward FLAGGED)
        direction: "positive" or "negative" contribution
        importance_rank: Rank among all features (1 = most important)
    """
    feature_name: str
    feature_value: Any
    shap_value: float
    direction: str = "positive"
    importance_rank: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "feature_value": self.feature_value,
            "shap_value": round(self.shap_value, 4),
            "direction": self.direction,
            "importance_rank": self.importance_rank
        }


@dataclass
class Explanation:
    """
    Complete explanation for a scoring decision.
    
    Contains SHAP values, feature contributions, and human summary.
    """
    verdict: Verdict
    confidence: float
    
    # Feature contributions
    contributions: List[FeatureContribution] = field(default_factory=list)
    
    # SHAP values for each class
    shap_values_fair: List[float] = field(default_factory=list)
    shap_values_inflated: List[float] = field(default_factory=list)
    shap_values_flagged: List[float] = field(default_factory=list)
    
    # Base value (expected value)
    base_value: float = 0.0
    
    # Human-readable explanations
    summary: str = ""
    detailed_explanation: str = ""
    
    # Top factors
    top_positive_factors: List[str] = field(default_factory=list)
    top_negative_factors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON."""
        return {
            "verdict": self.verdict.value,
            "confidence": round(self.confidence, 3),
            "contributions": [c.to_dict() for c in self.contributions],
            "base_value": round(self.base_value, 4),
            "summary": self.summary,
            "detailed_explanation": self.detailed_explanation,
            "top_positive_factors": self.top_positive_factors,
            "top_negative_factors": self.top_negative_factors
        }


class ClaimExplainer:
    """
    Generates SHAP explanations for claim scoring decisions.
    
    USAGE:
        explainer = ClaimExplainer()
        
        # Get explanation for a scoring result
        explanation = explainer.explain(
            scoring_result,
            extracted_claim,
            rate_result
        )
        
        print(explanation.summary)
        for contrib in explanation.contributions:
            print(f"  {contrib.feature_name}: {contrib.shap_value:+.3f}")
    
    HOW SHAP WORKS:
    1. Create a "background" dataset of typical claims
    2. For each prediction, calculate how each feature shifts the output
    3. SHAP values are additive: base_value + sum(shap_values) = prediction
    """
    
    # Human-readable feature names
    FEATURE_LABELS = {
        "rate_deviation_percent": "Rate deviation from market (%)",
        "rate_deviation_amount": "Rate deviation (£)",
        "claimed_rate": "Claimed daily rate",
        "market_mean_rate": "Market average rate",
        "market_rate_count": "Number of comparable rates",
        "hire_duration_days": "Hire duration (days)",
        "extraction_confidence": "Document extraction confidence",
        "vehicle_group_encoded": "Vehicle group",
        "has_region_match": "Region matched",
        "has_sufficient_comparables": "Sufficient comparable data"
    }
    
    def __init__(self, scorer: Optional[ClaimScorer] = None):
        """
        Initialize the explainer.
        
        Args:
            scorer: The scorer to explain (default: global scorer)
        """
        self.scorer = scorer or get_scorer()
        
        logger.info("ClaimExplainer initialized (RAG-based)")

    def explain(
        self,
        scoring_result: ScoringResult,
        claim: Optional[ExtractedClaim] = None,
        rate_result: Optional[RateMatchResult] = None
    ) -> Explanation:
        """
        Generate explanation for a scoring result.
        
        This uses the reasoning and evidence from RAG-based scoring.
        
        Args:
            scoring_result: The scoring result to explain
            claim: Original claim (for context)
            rate_result: Rate matching result (for context)
            
        Returns:
            Explanation with reasoning and evidence highlights
        """
        explanation = Explanation(
            verdict=scoring_result.verdict,
            confidence=scoring_result.confidence
        )
        
        # Build explanation from RAG reasoning and evidence
        explanation = self._build_rag_explanation(scoring_result)
        
        # Add context from claim and rate result
        explanation = self._add_context(explanation, claim, rate_result)
        
        return explanation
    
    def _build_rag_explanation(self, scoring_result: ScoringResult) -> Explanation:
        """
        Build explanation from RAG-based scoring results.
        
        Uses LLM reasoning and evidence directly instead of SHAP.
        """
        explanation = Explanation(
            verdict=scoring_result.verdict,
            confidence=scoring_result.confidence
        )
        
        # Use LLM reasoning from scoring result
        if scoring_result.reasoning and scoring_result.reasoning.get('llm_analysis'):
            explanation.summary = scoring_result.reasoning['llm_analysis']
        
        # Build feature contributions from evidence
        contributions = []
        if scoring_result.evidence_sources:
            for idx, source in enumerate(scoring_result.evidence_sources, 1):
                contribution = FeatureContribution(
                    feature=f"Market Rate #{idx}",
                    value=source.daily_rate,
                    importance=source.similarity_score,  # Use similarity as importance
                    direction="positive" if source.impact_on_verdict == "supports" else "negative"
                )
                contributions.append(contribution)
        
        explanation.feature_contributions = contributions
        
        # Add key features comparison
        if scoring_result.reasoning:
            features = scoring_result.reasoning.get('key_features', {})
            explanation.summary = (
                f"{explanation.summary or ''} "
                f"Claimed rate: £{features.get('claimed_rate', 'N/A')}/day, "
                f"Market average: £{features.get('market_mean_rate', 'N/A')}/day"
            ).strip()
        
        return explanation
    
    def _build_fallback_explanation(
        self,
        scoring_result: ScoringResult
    ) -> Explanation:
        """
        Build explanation without SHAP (using model feature importance).
        """
        feature_importance = scoring_result.feature_importance
        features = scoring_result.features
        
        contributions = []
        
        if feature_importance:
            for name, importance in feature_importance.items():
                value = getattr(features, name, None)
                contributions.append(FeatureContribution(
                    feature_name=self.FEATURE_LABELS.get(name, name),
                    feature_value=self._format_feature_value(name, value),
                    shap_value=importance,  # Use importance as proxy
                    direction="positive" if importance > 0 else "neutral"
                ))
        
        contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)
        
        for i, contrib in enumerate(contributions):
            contrib.importance_rank = i + 1
        
        summary = self._generate_summary(
            scoring_result.verdict,
            contributions,
            scoring_result.confidence
        )
        
        return Explanation(
            verdict=scoring_result.verdict,
            confidence=scoring_result.confidence,
            contributions=contributions,
            summary=summary,
            detailed_explanation=scoring_result.explanation
        )
    
    def _format_feature_value(self, name: str, value: Any) -> str:
        """
        Format a feature value for display.
        """
        if value is None:
            return "N/A"
        
        if name in ["claimed_rate", "market_mean_rate", "rate_deviation_amount"]:
            return f"£{float(value):.2f}"
        elif name == "rate_deviation_percent":
            return f"{float(value):+.1f}%"
        elif name == "extraction_confidence":
            return f"{float(value):.0%}"
        elif name == "vehicle_group_encoded":
            groups = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
            idx = int(value) if isinstance(value, (int, float)) else 0
            return groups[idx] if 0 <= idx < len(groups) else str(value)
        elif name in ["has_region_match", "has_sufficient_comparables"]:
            return "Yes" if value else "No"
        else:
            return str(value)
    
    def _generate_summary(
        self,
        verdict: Verdict,
        contributions: List[FeatureContribution],
        confidence: float
    ) -> str:
        """
        Generate a one-paragraph summary of the decision.
        """
        # Verdict description
        if verdict == Verdict.FAIR:
            verdict_text = "The claim appears to be fair and within market norms."
        elif verdict == Verdict.POTENTIALLY_INFLATED:
            verdict_text = "The claim shows signs of being above typical market rates."
        elif verdict == Verdict.FLAGGED:
            verdict_text = "The claim has been flagged for significant concerns."
        else:
            verdict_text = "Unable to make a reliable determination."
        
        # Top factors
        top_factors = contributions[:3]
        if top_factors:
            factor_texts = []
            for factor in top_factors:
                direction = "increased" if factor.shap_value > 0 else "decreased"
                factor_texts.append(
                    f"{factor.feature_name} ({factor.feature_value}) {direction} the score"
                )
            
            factors_text = "Key factors: " + "; ".join(factor_texts) + "."
        else:
            factors_text = ""
        
        return f"{verdict_text} (Confidence: {confidence:.0%}) {factors_text}"
    
    def _generate_detailed_explanation(
        self,
        verdict: Verdict,
        contributions: List[FeatureContribution]
    ) -> str:
        """
        Generate detailed explanation with all factors.
        """
        lines = [f"Verdict: {verdict.value}", "", "Factor Analysis:"]
        
        for contrib in contributions:
            sign = "+" if contrib.shap_value > 0 else ""
            lines.append(
                f"  {contrib.importance_rank}. {contrib.feature_name}: "
                f"{contrib.feature_value} "
                f"(contribution: {sign}{contrib.shap_value:.4f})"
            )
        
        return "\n".join(lines)
    
    def _add_context(
        self,
        explanation: Explanation,
        claim: Optional[ExtractedClaim],
        rate_result: Optional[RateMatchResult]
    ) -> Explanation:
        """
        Add contextual information to the explanation.
        """
        context_parts = []
        
        if claim:
            if claim.hire_company:
                context_parts.append(f"Hire company: {claim.hire_company}")
            if claim.vehicle.make and claim.vehicle.model:
                context_parts.append(
                    f"Vehicle: {claim.vehicle.make} {claim.vehicle.model}"
                )
        
        if rate_result:
            if rate_result.statistics.count > 0:
                context_parts.append(
                    f"Compared against {rate_result.statistics.count} market rates "
                    f"(range: £{rate_result.statistics.min_rate:.2f} - "
                    f"£{rate_result.statistics.max_rate:.2f}/day)"
                )
        
        if context_parts:
            explanation.summary += " " + " | ".join(context_parts)
        
        return explanation


# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------
_default_explainer: Optional[ClaimExplainer] = None


def get_explainer() -> ClaimExplainer:
    """Get the default explainer (singleton)."""
    global _default_explainer
    
    if _default_explainer is None:
        _default_explainer = ClaimExplainer()
    
    return _default_explainer


def explain_score(
    scoring_result: ScoringResult,
    claim: Optional[ExtractedClaim] = None,
    rate_result: Optional[RateMatchResult] = None
) -> Explanation:
    """
    Quick function to explain a scoring result.
    
    USAGE:
        from explainer import explain_score
        
        explanation = explain_score(scoring_result)
        print(explanation.summary)
    """
    explainer = get_explainer()
    return explainer.explain(scoring_result, claim, rate_result)