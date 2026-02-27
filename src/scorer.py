"""
scorer.py - ML Claim Scoring Model

WHY THIS FILE EXISTS:
- Classify claims as FAIR, POTENTIALLY_INFLATED, or FLAGGED
- Use traditional ML (scikit-learn) for interpretable predictions
- Combine multiple signals into a single score
- Provide probability estimates for confidence

HOW THE SCORER WORKS:
1. Takes features from extraction and rate matching
2. Runs through trained classifier
3. Returns verdict + probability + feature importance

MODEL ARCHITECTURE:
- Algorithm: Gradient Boosting Classifier (or Random Forest)
- Input: Numerical features (rate deviation, duration, confidence, etc.)
- Output: Class probabilities for each verdict

WHY GRADIENT BOOSTING:
- Handles non-linear relationships
- Good with small datasets
- Interpretable with SHAP
- Fast inference (important for real-time)

FEATURES USED:
- Rate deviation from market (%)
- Rate deviation from market (£)
- Hire duration (days)
- Extraction confidence
- Vehicle group (encoded)
- Has region match (binary)
- Number of comparable rates found
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

from extractor import ExtractedClaim
from rate_matcher import RateMatchResult, RateComparison
from config import settings, get_absolute_path

# Set up logging
logger = logging.getLogger(__name__)


class Verdict(Enum):
    """
    Possible verdicts for a claim.
    
    FAIR: Claim appears legitimate, rate is within market norms
    POTENTIALLY_INFLATED: Rate is above market, warrants review
    FLAGGED: Significant concerns, likely inflated or fraudulent
    INSUFFICIENT_DATA: Cannot make determination
    """
    FAIR = "FAIR"
    POTENTIALLY_INFLATED = "POTENTIALLY_INFLATED"
    FLAGGED = "FLAGGED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class ScoringFeatures:
    """
    Features extracted for ML scoring.
    
    These are the inputs to the classifier.
    """
    # Rate comparison features
    rate_deviation_percent: float = 0.0
    rate_deviation_amount: float = 0.0
    claimed_rate: float = 0.0
    market_mean_rate: float = 0.0
    market_rate_count: int = 0
    
    # Claim features
    hire_duration_days: int = 0
    extraction_confidence: float = 0.0
    
    # Vehicle features (encoded)
    vehicle_group_encoded: int = 0
    
    # Match quality
    has_region_match: bool = False
    has_sufficient_comparables: bool = False
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.rate_deviation_percent,
            self.rate_deviation_amount,
            self.claimed_rate,
            self.market_mean_rate,
            self.market_rate_count,
            self.hire_duration_days,
            self.extraction_confidence,
            self.vehicle_group_encoded,
            1.0 if self.has_region_match else 0.0,
            1.0 if self.has_sufficient_comparables else 0.0
        ]).reshape(1, -1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rate_deviation_percent": self.rate_deviation_percent,
            "rate_deviation_amount": self.rate_deviation_amount,
            "claimed_rate": self.claimed_rate,
            "market_mean_rate": self.market_mean_rate,
            "market_rate_count": self.market_rate_count,
            "hire_duration_days": self.hire_duration_days,
            "extraction_confidence": self.extraction_confidence,
            "vehicle_group_encoded": self.vehicle_group_encoded,
            "has_region_match": self.has_region_match,
            "has_sufficient_comparables": self.has_sufficient_comparables
        }
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get list of feature names (for SHAP)."""
        return [
            "rate_deviation_percent",
            "rate_deviation_amount",
            "claimed_rate",
            "market_mean_rate",
            "market_rate_count",
            "hire_duration_days",
            "extraction_confidence",
            "vehicle_group_encoded",
            "has_region_match",
            "has_sufficient_comparables"
        ]


@dataclass
class ScoringResult:
    """
    Result of scoring a claim.
    
    Contains verdict, probabilities, and feature importance.
    """
    verdict: Verdict
    confidence: float  # Probability of the predicted class
    probabilities: Dict[str, float] = field(default_factory=dict)
    features: ScoringFeatures = field(default_factory=ScoringFeatures)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON."""
        return {
            "verdict": self.verdict.value,
            "confidence": round(self.confidence, 3),
            "probabilities": {k: round(v, 3) for k, v in self.probabilities.items()},
            "features": self.features.to_dict(),
            "feature_importance": {k: round(v, 4) for k, v in self.feature_importance.items()},
            "explanation": self.explanation
        }


class ClaimScorer:
    """
    ML-based claim scoring model.
    
    USAGE:
        scorer = ClaimScorer()
        
        # Score a claim
        result = scorer.score(extracted_claim, rate_match_result)
        
        print(f"Verdict: {result.verdict.value}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Explanation: {result.explanation}")
    
    TRAINING:
        # Train on labeled data
        scorer.train(features_df, labels)
        scorer.save_model("models/claim_classifier.pkl")
    """
    
    # Class labels (must match training data)
    CLASSES = [Verdict.FAIR, Verdict.POTENTIALLY_INFLATED, Verdict.FLAGGED]
    CLASS_NAMES = ["FAIR", "POTENTIALLY_INFLATED", "FLAGGED"]
    
    # Vehicle group encoding
    VEHICLE_GROUPS = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the scorer.
        
        Args:
            model_path: Path to saved model file (optional)
        """
        self.model: Optional[GradientBoostingClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained: bool = False
        
        # Try to load existing model
        if model_path:
            self.load_model(model_path)
        else:
            default_path = get_absolute_path(settings.models_dir / "claim_classifier.pkl")
            if default_path.exists():
                self.load_model(default_path)
            else:
                # Initialize with default model (rule-based fallback)
                logger.info("No trained model found, using rule-based scoring")
                self._init_default_model()
        
        logger.info(f"ClaimScorer initialized (trained={self.is_trained})")
    
    def _init_default_model(self):
        """
        Initialize a default model when no trained model exists.
        
        Creates a simple model trained on synthetic data.
        This allows the system to work out-of-the-box.
        """
        # Create synthetic training data
        # This represents typical patterns we expect to see
        np.random.seed(42)
        n_samples = 300
        
        # Generate synthetic features
        X = []
        y = []
        
        # FAIR claims: low deviation, good data
        for _ in range(100):
            X.append([
                np.random.uniform(-10, 10),   # deviation %
                np.random.uniform(-5, 5),     # deviation £
                np.random.uniform(40, 60),    # claimed rate
                np.random.uniform(45, 55),    # market mean
                np.random.randint(5, 20),     # rate count
                np.random.randint(3, 21),     # duration
                np.random.uniform(0.7, 1.0),  # confidence
                np.random.randint(0, 9),      # vehicle group
                np.random.choice([0, 1]),     # region match
                1                             # sufficient comparables
            ])
            y.append(0)  # FAIR
        
        # POTENTIALLY_INFLATED: moderate deviation
        for _ in range(100):
            X.append([
                np.random.uniform(15, 35),    # deviation %
                np.random.uniform(8, 20),     # deviation £
                np.random.uniform(60, 85),    # claimed rate
                np.random.uniform(45, 55),    # market mean
                np.random.randint(3, 15),     # rate count
                np.random.randint(5, 30),     # duration
                np.random.uniform(0.5, 0.9),  # confidence
                np.random.randint(0, 9),      # vehicle group
                np.random.choice([0, 1]),     # region match
                1                             # sufficient comparables
            ])
            y.append(1)  # POTENTIALLY_INFLATED
        
        # FLAGGED: high deviation, suspicious patterns
        for _ in range(100):
            X.append([
                np.random.uniform(40, 100),   # deviation %
                np.random.uniform(20, 60),    # deviation £
                np.random.uniform(85, 150),   # claimed rate
                np.random.uniform(45, 55),    # market mean
                np.random.randint(1, 10),     # rate count
                np.random.randint(7, 45),     # duration
                np.random.uniform(0.3, 0.7),  # confidence
                np.random.randint(0, 9),      # vehicle group
                np.random.choice([0, 1]),     # region match
                np.random.choice([0, 1])      # sufficient comparables
            ])
            y.append(2)  # FLAGGED
        
        X = np.array(X)
        y = np.array(y)
        
        # Train model
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        
        self.is_trained = True
        logger.info("Default model trained on synthetic data")
    
    def extract_features(
        self,
        claim: ExtractedClaim,
        rate_result: RateMatchResult
    ) -> ScoringFeatures:
        """
        Extract scoring features from claim and rate match result.
        
        This transforms the raw data into numerical features
        that the ML model can process.
        """
        # Vehicle group encoding
        group = claim.vehicle.group or "C"  # Default to C if unknown
        try:
            group_encoded = self.VEHICLE_GROUPS.index(group.upper())
        except (ValueError, AttributeError):
            group_encoded = 2  # Default to C (index 2)
        
        return ScoringFeatures(
            rate_deviation_percent=rate_result.deviation_percent,
            rate_deviation_amount=rate_result.deviation_amount,
            claimed_rate=rate_result.claimed_rate,
            market_mean_rate=rate_result.statistics.mean_rate,
            market_rate_count=rate_result.statistics.count,
            hire_duration_days=claim.hire_period.duration_days or 0,
            extraction_confidence=claim.extraction_confidence,
            vehicle_group_encoded=group_encoded,
            has_region_match=bool(claim.accident_location),
            has_sufficient_comparables=rate_result.has_sufficient_data
        )
    
    def score(
        self,
        claim: ExtractedClaim,
        rate_result: RateMatchResult
    ) -> ScoringResult:
        """
        Score a claim and return verdict with explanation.
        
        This is the MAIN METHOD.
        
        Args:
            claim: Extracted claim data
            rate_result: Rate matching result
            
        Returns:
            ScoringResult with verdict and confidence
        """
        # Check for insufficient data
        if not rate_result.has_sufficient_data:
            return ScoringResult(
                verdict=Verdict.INSUFFICIENT_DATA,
                confidence=0.0,
                explanation="Not enough comparable market rates to make a reliable assessment."
            )
        
        # Extract features
        features = self.extract_features(claim, rate_result)
        
        # Get model prediction
        if self.model is not None and self.scaler is not None:
            # Scale features
            X = features.to_array()
            X_scaled = self.scaler.transform(X)
            
            # Predict probabilities
            proba = self.model.predict_proba(X_scaled)[0]
            predicted_class = self.model.predict(X_scaled)[0]
            
            # Map to verdict
            verdict = self.CLASSES[predicted_class]
            confidence = proba[predicted_class]
            
            # Get all probabilities
            probabilities = {
                self.CLASS_NAMES[i]: proba[i]
                for i in range(len(self.CLASS_NAMES))
            }
            
            # Get feature importance
            feature_importance = self._get_feature_importance(features)
            
        else:
            # Fallback to rule-based scoring
            verdict, confidence, probabilities = self._rule_based_score(features)
            feature_importance = {}
        
        # Generate explanation
        explanation = self._generate_explanation(
            verdict, features, rate_result
        )
        
        return ScoringResult(
            verdict=verdict,
            confidence=confidence,
            probabilities=probabilities,
            features=features,
            feature_importance=feature_importance,
            explanation=explanation
        )
    
    def _rule_based_score(
        self,
        features: ScoringFeatures
    ) -> Tuple[Verdict, float, Dict[str, float]]:
        """
        Fallback rule-based scoring when no trained model.
        
        Uses simple thresholds based on rate deviation.
        """
        deviation = features.rate_deviation_percent
        
        if deviation <= 15:
            verdict = Verdict.FAIR
            confidence = 1.0 - (deviation / 15) * 0.3  # 0.7-1.0
        elif deviation <= 40:
            verdict = Verdict.POTENTIALLY_INFLATED
            confidence = 0.6 + (deviation - 15) / 25 * 0.3  # 0.6-0.9
        else:
            verdict = Verdict.FLAGGED
            confidence = min(0.95, 0.7 + (deviation - 40) / 60 * 0.25)
        
        # Build probabilities (rough estimates)
        if verdict == Verdict.FAIR:
            proba = {"FAIR": confidence, "POTENTIALLY_INFLATED": 0.2, "FLAGGED": 0.1}
        elif verdict == Verdict.POTENTIALLY_INFLATED:
            proba = {"FAIR": 0.15, "POTENTIALLY_INFLATED": confidence, "FLAGGED": 0.2}
        else:
            proba = {"FAIR": 0.05, "POTENTIALLY_INFLATED": 0.15, "FLAGGED": confidence}
        
        # Normalize
        total = sum(proba.values())
        proba = {k: v / total for k, v in proba.items()}
        
        return verdict, confidence, proba
    
    def _get_feature_importance(
        self,
        features: ScoringFeatures
    ) -> Dict[str, float]:
        """
        Get feature importance from the model.
        
        Shows which features most influenced this prediction.
        """
        if self.model is None:
            return {}
        
        # Get global feature importance from model
        importance = self.model.feature_importances_
        feature_names = features.feature_names()
        
        return {
            name: float(imp)
            for name, imp in zip(feature_names, importance)
        }
    
    def _generate_explanation(
        self,
        verdict: Verdict,
        features: ScoringFeatures,
        rate_result: RateMatchResult
    ) -> str:
        """
        Generate human-readable explanation for the verdict.
        
        This is crucial for transparency and trust.
        """
        parts = []
        
        # Verdict summary
        if verdict == Verdict.FAIR:
            parts.append("This claim appears to be within normal market parameters.")
        elif verdict == Verdict.POTENTIALLY_INFLATED:
            parts.append("This claim shows signs of being above market rates.")
        elif verdict == Verdict.FLAGGED:
            parts.append("This claim has significant concerns requiring review.")
        else:
            parts.append("Unable to make a reliable assessment.")
        
        # Rate comparison
        if rate_result.statistics.count > 0:
            parts.append(
                f"The claimed rate of £{features.claimed_rate:.2f}/day "
                f"is {abs(features.rate_deviation_percent):.1f}% "
                f"{'above' if features.rate_deviation_percent > 0 else 'below'} "
                f"the market average of £{features.market_mean_rate:.2f}/day "
                f"(based on {features.market_rate_count} comparable rates)."
            )
        
        # Market range
        if rate_result.statistics.count >= 3:
            parts.append(
                f"Market rates range from £{rate_result.statistics.min_rate:.2f} "
                f"to £{rate_result.statistics.max_rate:.2f}/day."
            )
        
        # Confidence note
        if features.extraction_confidence < 0.7:
            parts.append(
                "Note: Document extraction confidence is low, "
                "some fields may be inaccurate."
            )
        
        return " ".join(parts)
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the model on labeled data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0=FAIR, 1=POTENTIALLY_INFLATED, 2=FLAGGED)
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary with training metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        
        logger.info(f"Model trained. Test accuracy: {accuracy:.2%}")
        
        return {
            "accuracy": accuracy,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_importance": dict(zip(
                ScoringFeatures.feature_names(),
                self.model.feature_importances_.tolist()
            ))
        }
    
    def save_model(self, path: Optional[Path] = None) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Where to save (default: models/claim_classifier.pkl)
        """
        if path is None:
            path = get_absolute_path(settings.models_dir / "claim_classifier.pkl")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "is_trained": self.is_trained
        }
        
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model file
        """
        path = Path(path)
        
        if not path.exists():
            logger.warning(f"Model file not found: {path}")
            return
        
        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)
            
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.is_trained = model_data.get("is_trained", True)
            
            logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")


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
    rate_result: RateMatchResult
) -> ScoringResult:
    """
    Quick function to score a claim.
    
    USAGE:
        from scorer import score_claim
        
        result = score_claim(claim, rate_match)
        print(result.verdict.value)
    """
    scorer = get_scorer()
    return scorer.score(claim, rate_result)