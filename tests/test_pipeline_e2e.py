"""
End-to-End Pipeline Test

Tests the complete flow:
PDF → Parse → Extract → Match → Score → Explain → JSON Result
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import ClaimsPipeline
from src.config import settings


def test_pipeline_with_sample():
    """Test full pipeline with a real PDF."""
    print("=" * 80)
    print("📋 CLAIMNOW END-TO-END PIPELINE TEST")
    print("=" * 80)
    print()
    
    # Find a sample PDF if it exists
    pdf_path = Path(__file__).parent.parent / "data" / "court_pack_001_LDS320_2023.pdf"
    
    if not pdf_path.exists():
        print(f"❌ Sample PDF not found at {pdf_path}")
        print(f"   Place a PDF in: data/ folder")
        return False
    
    print(f"✅ Found sample PDF: {pdf_path.name}")
    print()
    
    # Create pipeline
    print("🔧 Initializing pipeline...")
    pipeline = ClaimsPipeline(verbose=True)
    print("   ✓ Pipeline ready")
    print()
    
    # Analyze
    print("🚀 Starting analysis...")
    print()
    
    try:
        result = pipeline.analyze(pdf_path, claim_id="TEST-001")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Display results
    print()
    print("=" * 80)
    print("📊 ANALYSIS RESULTS")
    print("=" * 80)
    print()
    
    print(f"Claim ID:          {result.claim_id}")
    print(f"Filename:          {result.filename}")
    print(f"Analyzed at:       {result.analyzed_at}")
    print()
    
    print(f"Verdict:           {result.verdict.value}")
    print(f"Confidence:        {result.confidence:.1%}")
    print(f"Summary:           {result.summary}")
    print()
    
    print(f"Total time:        {result.total_duration_ms:.0f} ms")
    print(f"Stages completed:  {result.completed_stages}/5")
    
    if result.failed_stage:
        print(f"❌ Failed at:       {result.failed_stage}")
    else:
        print(f"✅ All stages passed!")
    
    print()
    print("📈 Stage Timing:")
    for stage_name, stage_result in result.stages.items():
        status = "✓" if stage_result.success else "✗"
        print(f"   {status} {stage_name:15} {stage_result.duration_ms:8.1f} ms", end="")
        if stage_result.error:
            print(f" - {stage_result.error}")
        else:
            print()
    
    print()
    
    # Detailed results
    if result.extracted_claim:
        print("🚗 EXTRACTED CLAIM:")
        print(f"   Vehicle:     {result.extracted_claim.vehicle.make} {result.extracted_claim.vehicle.model}")
        print(f"   Group:       {result.extracted_claim.vehicle.group}")
        if result.extracted_claim.hire_period.start_date:
            print(f"   Start:       {result.extracted_claim.hire_period.start_date}")
        if result.extracted_claim.hire_period.end_date:
            print(f"   End:         {result.extracted_claim.hire_period.end_date}")
        if result.extracted_claim.hire_period.duration_days:
            print(f"   Duration:    {result.extracted_claim.hire_period.duration_days} days")
        if result.extracted_claim.rate.daily_rate:
            print(f"   Daily rate:  £{result.extracted_claim.rate.daily_rate:.2f}")
        print()
    
    if result.rate_match:
        print("📊 MARKET COMPARISON:")
        print(f"   Rates found: {result.rate_match.statistics.count}")
        print(f"   Min rate:    £{result.rate_match.statistics.min_rate:.2f}")
        print(f"   Mean rate:   £{result.rate_match.statistics.mean_rate:.2f}")
        print(f"   Max rate:    £{result.rate_match.statistics.max_rate:.2f}")
        if result.rate_match.deviation_percent:
            print(f"   Deviation:   {result.rate_match.deviation_percent:+.1f}%")
        print()
    
    if result.scoring:
        print("✅ SCORING RESULT:")
        print(f"   Verdict:     {result.scoring.verdict.value}")
        print(f"   Confidence:  {result.scoring.confidence:.1%}")
        print(f"   Reasoning:   {result.scoring.reasoning[:100]}...")
        print(f"   Evidence:    {len(result.scoring.evidence_sources)} sources")
        
        if result.scoring.evidence_sources:
            print()
            print("   Evidence details:")
            for ev in result.scoring.evidence_sources[:3]:
                print(f"      • {ev.source_id}: £{ev.daily_rate:.2f}/day (Group {ev.vehicle_group}, {ev.similarity_score:.1%} match)")
        
        # Show audit log
        print()
        print("   Audit Trail:")
        print(f"      ID:       {result.scoring.audit_log.audit_id}")
        print(f"      Time:     {result.scoring.audit_log.timestamp}")
        print(f"      Method:   {result.scoring.audit_log.scoring_method}")
        print(f"      Latency:  {result.scoring.audit_log.decision_latency_ms:.1f} ms")
        print()
    
    # Show structured JSON
    print("📄 STRUCTURED JSON OUTPUT:")
    print()
    import json
    json_output = result.to_dict()
    print(json.dumps(json_output, indent=2)[:500] + "...")
    print()
    
    return True


if __name__ == "__main__":
    success = test_pipeline_with_sample()
    sys.exit(0 if success else 1)
