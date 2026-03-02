"""
API Integration Test

Tests the FastAPI server endpoints.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_api_endpoints():
    """Test API endpoints via direct imports."""
    print("=" * 80)
    print("🌐 API ENDPOINT TESTS")
    print("=" * 80)
    print()
    
    try:
        from src.main import app
        from fastapi.testclient import TestClient
    except ImportError:
        print("⚠️  FastAPI TestClient not available")
        print("   Install: pip install fastapi[all]")
        return False
    
    # Create test client
    client = TestClient(app)
    
    # Test 1: Health check
    print("1️⃣  Health Check Endpoint")
    response = client.get("/health")
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Status: {data.get('status')}")
        print(f"   Version: {data.get('version')}")
        print(f"   Ollama: {data.get('ollama_available')}")
        print(f"   ✓ Health check passed")
    else:
        print(f"   ✗ Failed: {response.text}")
        return False
    
    print()
    
    # Test 2: Analyze text endpoint
    print("2️⃣  Analyze Text Endpoint")
    
    test_claim = """
    CREDIT HIRE INVOICE
    
    Vehicle: BMW 320d Group D
    Daily Rate: £65.00
    Hire Period: 14 days (01/01/2024 - 14/01/2024)
    Total Cost: £910.00
    
    Claimant: John Doe
    Accident Date: 31/12/2023
    """
    
    payload = {
        "text": test_claim,
        "claim_id": "API-TEST-001"
    }
    
    response = client.post("/api/analyze-text", json=payload)
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"   Verdict: {result.get('verdict')}")
        print(f"   Confidence: {result.get('confidence')}")
        print(f"   ✓ Text analysis passed")
        
        # Show structured response
        print()
        print("   Response structure:")
        print(f"      - verdict: {type(result.get('verdict'))}")
        print(f"      - reasoning: {type(result.get('reasoning'))}")
        print(f"      - evidence: {type(result.get('evidence'))}")
        print(f"      - audit: {type(result.get('audit'))}")
        
    else:
        print(f"   ✗ Failed: {response.status_code}")
        print(f"   {response.text}")
        return False
    
    print()
    
    # Test 3: RAG Query endpoint
    print("3️⃣  RAG Query Endpoint")
    
    rag_query = "What are typical rates for Group C vehicles in London?"
    
    response = client.post("/api/rag-query", json={"query": rag_query})
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ RAG query processed")
        print(f"   Answer: {result.get('answer', '')[:100]}...")
    else:
        print(f"   ⚠️  RAG not available: {response.status_code}")
    
    print()
    print("✅ API tests completed")
    return True


def test_direct_analysis():
    """Test analysis without API."""
    print("\n" + "=" * 80)
    print("🔬 DIRECT ANALYSIS TEST")
    print("=" * 80)
    print()
    
    from src.document_parser import DocumentContent, PageContent
    from src.extractor import ExtractedClaim
    from src.pipeline import ClaimsPipeline
    
    # Create pipeline
    pipeline = ClaimsPipeline()
    
    # Test with raw text
    test_text = """
    INVOICE FOR VEHICLE HIRE
    
    Vehicle Make: BMW
    Vehicle Model: 3 Series
    Vehicle Group: D
    
    Hire Start Date: 1 January 2024
    Hire End Date: 15 January 2024
    
    Daily Rate: £70
    Total Days: 14
    Total Cost: £980
    
    Accident Date: 31 December 2023
    Location: London
    """
    
    print("Test claim:")
    print(test_text)
    print()
    
    result = pipeline.analyze_text(test_text, filename="test.txt", claim_id="DIRECT-001")
    
    print(f"✓ Analysis completed")
    print(f"  Verdict: {result.verdict.value}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Complete: {result.is_complete}")
    print()
    
    return True


if __name__ == "__main__":
    try:
        # Run direct analysis first (doesn't require API)
        test_direct_analysis()
        
        # Try API tests
        test_api_endpoints()
        
        print("\n" + "=" * 80)
        print("✅ All tests completed")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
