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

import numpy as np
import shap

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
        self._shap_explainer: Optional[shap.Explainer] = None
        
        # Initialize SHAP explainer if model is available
        if self.scorer.model is not None:
            self._init_shap_explainer()
        
        logger.info("ClaimExplainer initialized")
    
    def _init_shap_explainer(self):
        """
        Initialize the SHAP explainer with background data.
        
        SHAP needs a "background" dataset to compute expected values.
        We create synthetic background data representing typical claims.
        """
        try:
            # Create background data (100 samples)
            np.random.seed(42)
            n_background = 100
            
            background_data = np.array([
                [
                    np.random.uniform(-20, 50),   # rate_deviation_percent
                    np.random.uniform(-10, 30),   # rate_deviation_amount
                    np.random.uniform(40, 100),   # claimed_rate
                    np.random.uniform(40, 60),    # market_mean_rate
                    np.random.randint(3, 20),     # market_rate_count
                    np.random.randint(3, 30),     # hire_duration_days
                    np.random.uniform(0.5, 1.0),  # extraction_confidence
                    np.random.randint(0, 9),      # vehicle_group_encoded
                    np.random.choice([0, 1]),     # has_region_match
                    np.random.choice([0, 1])      # has_sufficient_comparables
                ]
                for _ in range(n_background)
            ])
            
            # Scale background data
            if self.scorer.scaler is not None:
                background_scaled = self.scorer.scaler.transform(background_data)
            else:
                background_scaled = background_data
            
            # Create SHAP explainer
            # TreeExplainer is optimized for tree-based models (like GradientBoosting)
            self._shap_explainer = shap.TreeExplainer(
                self.scorer.model,
                background_scaled,
                feature_perturbation="interventional"
            )
            
            logger.info("SHAP explainer initialized")
            
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
            self._shap_explainer = None
    
    def explain(
        self,
        scoring_result: ScoringResult,
        claim: Optional[ExtractedClaim] = None,
        rate_result: Optional[RateMatchResult] = None
    ) -> Explanation:
        """
        Generate explanation for a scoring result.
        
        This is the MAIN METHOD.
        
        Args:
            scoring_result: The scoring result to explain
            claim: Original claim (for context)
            rate_result: Rate matching result (for context)
            
        Returns:
            Explanation with SHAP values and human summary
        """
        explanation = Explanation(
            verdict=scoring_result.verdict,
            confidence=scoring_result.confidence
        )
        
        # Get SHAP values if explainer is available
        if self._shap_explainer is not None:
            try:
                shap_explanation = self._compute_shap_values(scoring_result.features)
                explanation = self._build_explanation(
                    shap_explanation,
                    scoring_result.features,
                    scoring_result.verdict,
                    scoring_result.confidence
                )
            except Exception as e:
                logger.warning(f"SHAP computation failed: {e}")
                # Fall back to feature importance from model
                explanation = self._build_fallback_explanation(
                    scoring_result
                )
        else:
            # No SHAP, use model feature importance
            explanation = self._build_fallback_explanation(scoring_result)
        
        # Add context from claim and rate result
        explanation = self._add_context(explanation, claim, rate_result)
        
        return explanation
    
    def _compute_shap_values(
        self,
        features: ScoringFeatures
    ) -> shap.Explanation:
        """
        Compute SHAP values for the given features.
        """
        # Convert features to array
        X = features.to_array()
        
        # Scale if scaler is available
        if self.scorer.scaler is not None:
            X_scaled = self.scorer.scaler.transform(X)
        else:
            X_scaled = X
        
        # Compute SHAP values
        shap_values = self._shap_explainer(X_scaled)
        
        return shap_values
    
    def _build_explanation(
        self,
        shap_explanation: shap.Explanation,
        features: ScoringFeatures,
        verdict: Verdict,
        confidence: float
    ) -> Explanation:
        """
        Build Explanation object from SHAP values.
        """
        # Get feature names and values
        feature_names = features.feature_names()
        feature_values = features.to_array()[0]
        
        # SHAP values are per-class for multi-class models
        # Shape: (n_samples, n_features, n_classes)
        shap_vals = shap_explanation.values[0]  # First sample
        
        # For the predicted class
        class_idx = self.scorer.CLASSES.index(verdict) if verdict in self.scorer.CLASSES else 0
        
        # Get SHAP values for predicted class
        if len(shap_vals.shape) > 1:
            # Multi-class: shap_vals is (n_features, n_classes)
            class_shap = shap_vals[:, class_idx]
        else:
            # Binary or single output
            class_shap = shap_vals
        
        # Build contributions
        contributions = []
        for i, (name, value, shap_val) in enumerate(zip(
            feature_names, feature_values, class_shap
        )):
            contributions.append(FeatureContribution(
                feature_name=self.FEATURE_LABELS.get(name, name),
                feature_value=self._format_feature_value(name, value),
                shap_value=float(shap_val),
                direction="positive" if shap_val > 0 else "negative"
            ))
        
        # Sort by absolute SHAP value
        contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)
        
        # Add importance rank
        for i, contrib in enumerate(contributions):
            contrib.importance_rank = i + 1
        
        # Get top positive and negative factors
        top_positive = [
            c.feature_name for c in contributions
            if c.shap_value > 0.01
        ][:3]
        
        top_negative = [
            c.feature_name for c in contributions
            if c.shap_value < -0.01
        ][:3]
        
        # Base value
        base_value = float(shap_explanation.base_values[0])
        if isinstance(base_value, np.ndarray):
            base_value = float(base_value[class_idx])
        
        # Generate summary
        summary = self._generate_summary(verdict, contributions, confidence)
        detailed = self._generate_detailed_explanation(verdict, contributions)
        
        return Explanation(
            verdict=verdict,
            confidence=confidence,
            contributions=contributions,
            base_value=base_value,
            summary=summary,
            detailed_explanation=detailed,
            top_positive_factors=top_positive,
            top_negative_factors=top_negative
        )
    
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