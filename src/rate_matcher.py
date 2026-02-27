"""
rate_matcher.py - Market Rate Comparison using RAG

WHY THIS FILE EXISTS:
- Compare claimed hire rates against market data
- Use RAG to find similar/comparable rates
- Calculate how the claim compares to market average
- Provide evidence for the scoring decision

HOW RATE MATCHING WORKS:
1. Take extracted claim data (vehicle group, region, dates)
2. Build a search query from claim attributes
3. Search vector store for similar rate records
4. Calculate statistics (min, max, average, percentile)
5. Determine if claimed rate is within normal range

WHAT MAKES RATES "COMPARABLE":
- Same vehicle group (most important)
- Same or similar region
- Similar time period (rates change over time)
- Same hire type (standard vs. prestige)

OUTPUT:
- List of comparable rates found
- Market statistics (average, range)
- Percentage deviation from market
- Recommendation based on comparison
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from statistics import mean, median, stdev
from enum import Enum

from vector_store import VectorStore, get_rates_store, SearchResult
from extractor import ExtractedClaim
from config import settings

# Set up logging
logger = logging.getLogger(__name__)


class RateComparison(Enum):
    """
    Result of comparing a rate to market data.
    
    BELOW_MARKET: Rate is lower than typical market rates
    WITHIN_MARKET: Rate is within normal market range (±15%)
    ABOVE_MARKET: Rate is 15-40% above market average
    SIGNIFICANTLY_ABOVE: Rate is >40% above market average
    INSUFFICIENT_DATA: Not enough comparable rates to assess
    """
    BELOW_MARKET = "below_market"
    WITHIN_MARKET = "within_market"
    ABOVE_MARKET = "above_market"
    SIGNIFICANTLY_ABOVE = "significantly_above"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class MarketRate:
    """
    A single market rate record.
    
    Represents one data point for comparison.
    """
    daily_rate: float
    vehicle_group: str
    region: Optional[str] = None
    company: Optional[str] = None
    year: Optional[int] = None
    source_id: str = ""
    similarity_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "daily_rate": self.daily_rate,
            "vehicle_group": self.vehicle_group,
            "region": self.region,
            "company": self.company,
            "year": self.year,
            "source_id": self.source_id,
            "similarity_score": self.similarity_score
        }


@dataclass
class MarketStatistics:
    """
    Statistical summary of market rates.
    
    Calculated from the comparable rates found.
    """
    count: int = 0                    # Number of rates found
    min_rate: float = 0.0             # Lowest rate
    max_rate: float = 0.0             # Highest rate
    mean_rate: float = 0.0            # Average rate
    median_rate: float = 0.0          # Middle value
    std_dev: float = 0.0              # Standard deviation
    percentile_25: float = 0.0        # 25th percentile
    percentile_75: float = 0.0        # 75th percentile
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "count": self.count,
            "min_rate": round(self.min_rate, 2),
            "max_rate": round(self.max_rate, 2),
            "mean_rate": round(self.mean_rate, 2),
            "median_rate": round(self.median_rate, 2),
            "std_dev": round(self.std_dev, 2),
            "percentile_25": round(self.percentile_25, 2),
            "percentile_75": round(self.percentile_75, 2)
        }


@dataclass
class RateMatchResult:
    """
    Complete result of rate matching.
    
    Contains comparable rates, statistics, and analysis.
    """
    # The claimed rate being analyzed
    claimed_rate: float
    vehicle_group: str
    
    # Comparable rates found
    comparable_rates: List[MarketRate] = field(default_factory=list)
    
    # Market statistics
    statistics: MarketStatistics = field(default_factory=MarketStatistics)
    
    # Analysis results
    comparison: RateComparison = RateComparison.INSUFFICIENT_DATA
    deviation_percent: float = 0.0  # How far from average (positive = above)
    deviation_amount: float = 0.0   # Difference in GBP
    
    # Search metadata
    search_query: str = ""
    filters_used: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON."""
        return {
            "claimed_rate": self.claimed_rate,
            "vehicle_group": self.vehicle_group,
            "comparable_rates": [r.to_dict() for r in self.comparable_rates],
            "statistics": self.statistics.to_dict(),
            "comparison": self.comparison.value,
            "deviation_percent": round(self.deviation_percent, 2),
            "deviation_amount": round(self.deviation_amount, 2),
            "search_query": self.search_query,
            "filters_used": self.filters_used
        }
    
    @property
    def has_sufficient_data(self) -> bool:
        """Check if we have enough data for reliable comparison."""
        return self.statistics.count >= 3
    
    @property
    def is_within_market(self) -> bool:
        """Check if rate is within normal market range."""
        return self.comparison == RateComparison.WITHIN_MARKET
    
    @property
    def is_above_market(self) -> bool:
        """Check if rate is above market average."""
        return self.comparison in [
            RateComparison.ABOVE_MARKET,
            RateComparison.SIGNIFICANTLY_ABOVE
        ]


class RateMatcher:
    """
    Matches claimed rates against market data using RAG.
    
    USAGE:
        matcher = RateMatcher()
        
        # From an extracted claim
        result = matcher.match_claim(extracted_claim)
        
        print(f"Market average: £{result.statistics.mean_rate}")
        print(f"Deviation: {result.deviation_percent}%")
        print(f"Comparison: {result.comparison.value}")
    
    OR:
        # Direct rate matching
        result = matcher.match_rate(
            daily_rate=65.0,
            vehicle_group="C",
            region="London"
        )
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        min_comparable_rates: int = 3
    ):
        """
        Initialize the rate matcher.
        
        Args:
            vector_store: Vector store with rate data (default: rates store)
            min_comparable_rates: Minimum rates needed for reliable comparison
        """
        self.vector_store = vector_store or get_rates_store()
        self.min_comparable_rates = min_comparable_rates
        
        logger.info("RateMatcher initialized")
    
    def match_claim(self, claim: ExtractedClaim) -> RateMatchResult:
        """
        Match rates for an extracted claim.
        
        This is the MAIN METHOD when processing documents.
        
        Args:
            claim: Extracted claim data from extractor.py
            
        Returns:
            RateMatchResult with comparison analysis
        """
        # Get daily rate from claim
        daily_rate = claim.rates.daily_rate
        
        if daily_rate is None:
            logger.warning("No daily rate in claim, cannot match")
            return RateMatchResult(
                claimed_rate=0.0,
                vehicle_group=claim.vehicle.group or "UNKNOWN",
                comparison=RateComparison.INSUFFICIENT_DATA
            )
        
        # Get vehicle group
        vehicle_group = claim.vehicle.group
        
        if vehicle_group is None:
            logger.warning("No vehicle group in claim, using general search")
            vehicle_group = "UNKNOWN"
        
        # Build region hint from claim if available
        region = claim.accident_location
        
        # Match the rate
        return self.match_rate(
            daily_rate=daily_rate,
            vehicle_group=vehicle_group,
            region=region
        )
    
    def match_rate(
        self,
        daily_rate: float,
        vehicle_group: str,
        region: Optional[str] = None,
        year: Optional[int] = None,
        top_k: int = 20
    ) -> RateMatchResult:
        """
        Match a rate against market data.
        
        Args:
            daily_rate: The claimed daily rate in GBP
            vehicle_group: Vehicle group (A-I)
            region: Geographic region (optional, improves matching)
            year: Year for temporal matching (optional)
            top_k: Number of comparable rates to retrieve
            
        Returns:
            RateMatchResult with full analysis
        """
        # Build search query
        query = self._build_search_query(vehicle_group, region, daily_rate)
        
        # Build metadata filters
        filters = self._build_filters(vehicle_group)
        
        # Search for comparable rates
        search_results = self.vector_store.search(
            query=query,
            top_k=top_k,
            where=filters if filters else None
        )
        
        # Convert to MarketRate objects
        comparable_rates = self._parse_search_results(search_results.results)
        
        # If strict filter returned too few results, try without filter
        if len(comparable_rates) < self.min_comparable_rates:
            logger.info("Few results with filters, trying broader search")
            search_results = self.vector_store.search(
                query=query,
                top_k=top_k * 2  # Get more results
            )
            comparable_rates = self._parse_search_results(search_results.results)
        
        # Calculate statistics
        statistics = self._calculate_statistics(comparable_rates)
        
        # Determine comparison result
        comparison, deviation_pct, deviation_amt = self._compare_rate(
            daily_rate, statistics
        )
        
        return RateMatchResult(
            claimed_rate=daily_rate,
            vehicle_group=vehicle_group,
            comparable_rates=comparable_rates,
            statistics=statistics,
            comparison=comparison,
            deviation_percent=deviation_pct,
            deviation_amount=deviation_amt,
            search_query=query,
            filters_used=filters
        )
    
    def _build_search_query(
        self,
        vehicle_group: str,
        region: Optional[str],
        daily_rate: float
    ) -> str:
        """
        Build a natural language search query.
        
        The query is embedded and used for similarity search.
        """
        parts = []
        
        # Vehicle group
        if vehicle_group and vehicle_group != "UNKNOWN":
            parts.append(f"Group {vehicle_group} vehicle")
        
        # Region
        if region:
            parts.append(f"in {region}")
        
        # Rate context
        parts.append(f"daily hire rate around £{daily_rate}")
        
        query = " ".join(parts)
        logger.debug(f"Search query: {query}")
        
        return query
    
    def _build_filters(self, vehicle_group: str) -> Dict[str, Any]:
        """
        Build metadata filters for the search.
        
        ChromaDB where filters for exact matching on metadata.
        """
        filters = {}
        
        if vehicle_group and vehicle_group != "UNKNOWN":
            filters["vehicle_group"] = vehicle_group
        
        return filters
    
    def _parse_search_results(
        self,
        results: List[SearchResult]
    ) -> List[MarketRate]:
        """
        Parse search results into MarketRate objects.
        
        Extracts rate and metadata from each result.
        """
        market_rates = []
        
        for result in results:
            # Try to get daily rate from metadata
            rate_value = result.metadata.get("daily_rate") or result.metadata.get("rate")
            
            if rate_value is None:
                # Try to extract from document text
                rate_value = self._extract_rate_from_text(result.document)
            
            if rate_value is not None:
                try:
                    market_rates.append(MarketRate(
                        daily_rate=float(rate_value),
                        vehicle_group=result.metadata.get("vehicle_group", ""),
                        region=result.metadata.get("region"),
                        company=result.metadata.get("company"),
                        year=result.metadata.get("year"),
                        source_id=result.id,
                        similarity_score=result.similarity
                    ))
                except (ValueError, TypeError):
                    logger.debug(f"Could not parse rate: {rate_value}")
        
        return market_rates
    
    def _extract_rate_from_text(self, text: str) -> Optional[float]:
        """
        Try to extract a rate value from document text.
        
        Fallback when rate isn't in metadata.
        """
        import re
        
        # Look for patterns like "£65", "65.00/day", "£55.50"
        patterns = [
            r'£(\d+\.?\d*)',          # £65 or £65.50
            r'(\d+\.?\d*)\s*/\s*day',  # 65/day
            r'daily.*?(\d+\.?\d*)',    # daily rate 65
            r'(\d+\.?\d*)\s*per\s*day' # 65 per day
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    def _calculate_statistics(
        self,
        rates: List[MarketRate]
    ) -> MarketStatistics:
        """
        Calculate statistical summary of market rates.
        """
        if not rates:
            return MarketStatistics()
        
        values = [r.daily_rate for r in rates]
        values_sorted = sorted(values)
        n = len(values)
        
        stats = MarketStatistics(
            count=n,
            min_rate=min(values),
            max_rate=max(values),
            mean_rate=mean(values),
            median_rate=median(values)
        )
        
        # Standard deviation (need at least 2 values)
        if n >= 2:
            stats.std_dev = stdev(values)
        
        # Percentiles
        if n >= 4:
            stats.percentile_25 = values_sorted[n // 4]
            stats.percentile_75 = values_sorted[(3 * n) // 4]
        else:
            stats.percentile_25 = stats.min_rate
            stats.percentile_75 = stats.max_rate
        
        return stats
    
    def _compare_rate(
        self,
        claimed_rate: float,
        stats: MarketStatistics
    ) -> tuple:
        """
        Compare claimed rate against market statistics.
        
        Returns:
            (RateComparison, deviation_percent, deviation_amount)
        """
        # Not enough data
        if stats.count < self.min_comparable_rates:
            return (RateComparison.INSUFFICIENT_DATA, 0.0, 0.0)
        
        # Calculate deviation from mean
        deviation_amount = claimed_rate - stats.mean_rate
        
        if stats.mean_rate > 0:
            deviation_percent = (deviation_amount / stats.mean_rate) * 100
        else:
            deviation_percent = 0.0
        
        # Determine comparison category
        # Using thresholds: ±15% = within market, 15-40% = above, >40% = significantly above
        if deviation_percent <= -15:
            comparison = RateComparison.BELOW_MARKET
        elif deviation_percent <= 15:
            comparison = RateComparison.WITHIN_MARKET
        elif deviation_percent <= 40:
            comparison = RateComparison.ABOVE_MARKET
        else:
            comparison = RateComparison.SIGNIFICANTLY_ABOVE
        
        return (comparison, deviation_percent, deviation_amount)
    
    def get_market_summary(
        self,
        vehicle_group: str,
        region: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get a summary of market rates for a vehicle group.
        
        Useful for reporting and dashboards.
        
        Args:
            vehicle_group: Vehicle group to summarize
            region: Optional region filter
            
        Returns:
            Dictionary with market summary
        """
        # Build query
        query = f"Group {vehicle_group} vehicle hire rates"
        if region:
            query += f" in {region}"
        
        # Search
        filters = {"vehicle_group": vehicle_group} if vehicle_group else None
        results = self.vector_store.search(query=query, top_k=50, where=filters)
        
        # Parse and calculate
        rates = self._parse_search_results(results.results)
        stats = self._calculate_statistics(rates)
        
        return {
            "vehicle_group": vehicle_group,
            "region": region,
            "statistics": stats.to_dict(),
            "sample_rates": [r.to_dict() for r in rates[:10]]
        }


# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------
_default_matcher: Optional[RateMatcher] = None


def get_rate_matcher() -> RateMatcher:
    """Get the default rate matcher (singleton)."""
    global _default_matcher
    
    if _default_matcher is None:
        _default_matcher = RateMatcher()
    
    return _default_matcher


def match_rate(
    daily_rate: float,
    vehicle_group: str,
    region: Optional[str] = None
) -> RateMatchResult:
    """
    Quick function to match a rate.
    
    USAGE:
        from rate_matcher import match_rate
        
        result = match_rate(65.0, "C", "London")
        print(f"Deviation: {result.deviation_percent}%")
    """
    matcher = get_rate_matcher()
    return matcher.match_rate(daily_rate, vehicle_group, region)