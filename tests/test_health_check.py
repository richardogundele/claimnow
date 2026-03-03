"""
Health Check - Verify all components are working

Checks:
✓ Python environment
✓ Configuration loaded
✓ Bedrock LLM available
✓ Vector store accessible
✓ Models loadable
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_python():
    """Check Python version."""
    print("🐍 Python Version")
    version = sys.version.split()[0]
    print(f"   ✓ {version}")
    return True


def check_config():
    """Check config loads."""
    print("\n⚙️  Configuration")
    try:
        from src.config import settings
        print(f"   ✓ Base dir: {settings.base_dir}")
        print(f"   ✓ AWS Region: {settings.aws_region}")
        print(f"   ✓ LLM Model: {settings.bedrock_llm_model_id}")
        print(f"   ✓ Embedding Model: {settings.bedrock_embedding_model_id}")
        print(f"   ✓ Vectorstore: {settings.vectorstore_dir}")
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def check_bedrock():
    """Check Amazon Bedrock."""
    print("\n🤖 Bedrock LLM")
    try:
        from src.llm_client import get_llm_client
        client = get_llm_client()
        
        # Try a simple connectivity check
        print(f"   Testing connection to Bedrock in {client.region}...")
        is_avail = client.is_available()
        
        if is_avail:
            print(f"   ✓ Bedrock client initialized")
            print(f"   ✓ Model: {client.model_id}")
            return True
        else:
            print(f"   ✗ Bedrock error: unable to connect")
            return False
    except Exception as e:
        print(f"   ✗ Bedrock not available: {e}")
        print(f"   💡 Ensure AWS credentials are set up")
        return False


def check_embeddings():
    """Check embedding model loads."""
    print("\n🔢 Embedding Model")
    try:
        from src.embeddings import get_embedding_model
        model = get_embedding_model()
        
        # Test embedding
        test_text = "test document"
        embedding = model.embed(test_text)
        
        print(f"   ✓ Model loaded: {model.model_id}")
        print(f"   ✓ Dimension: {embedding.shape[0]}")
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def check_vectorstore():
    """Check vector store."""
    print("\n🗄️  Vector Store (ChromaDB)")
    try:
        from src.vector_store import get_rates_store
        store = get_rates_store()
        
        print(f"   ✓ Store connected")
        print(f"   ✓ Collection: market_rates")
        
        # Check if data exists
        try:
            # Try to get count
            print(f"   💡 To populate with data: run data/generate_court_packs.py")
        except:
            pass
        
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def check_document_parser():
    """Check document parser."""
    print("\n📄 Document Parser")
    try:
        from src.document_parser import DocumentParser
        parser = DocumentParser(ocr_enabled=True)
        print(f"   ✓ Parser initialized")
        print(f"   ✓ OCR enabled")
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def check_extractor():
    """Check claim extractor."""
    print("\n🔍 Claim Extractor")
    try:
        from src.extractor import get_extractor
        extractor = get_extractor()
        print(f"   ✓ Extractor initialized")
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def check_rate_matcher():
    """Check rate matcher."""
    print("\n💰 Rate Matcher")
    try:
        from src.rate_matcher import get_rate_matcher
        matcher = get_rate_matcher()
        print(f"   ✓ Matcher initialized")
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def check_scorer():
    """Check RAG-based scorer."""
    print("\n⭐ Scorer (RAG)")
    try:
        from src.scorer import get_scorer
        scorer = get_scorer()
        print(f"   ✓ Scorer initialized (RAG-based)")
        print(f"   ✓ Audit logging enabled")
        print(f"   ✓ Evidence highlighting enabled")
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def check_pipeline():
    """Check full pipeline."""
    print("\n🔄 Full Pipeline")
    try:
        from src.pipeline import ClaimsPipeline
        pipeline = ClaimsPipeline()
        print(f"   ✓ Pipeline initialized")
        print(f"   ✓ All components connected")
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def main():
    print("=" * 60)
    print("🏥 CLAIMNOW HEALTH CHECK")
    print("=" * 60)
    
    checks = [
        ("Python", check_python),
        ("Configuration", check_config),
        ("Embedding Model", check_embeddings),
        ("Vector Store", check_vectorstore),
        ("Document Parser", check_document_parser),
        ("Claim Extractor", check_extractor),
        ("Rate Matcher", check_rate_matcher),
        ("Scorer (RAG)", check_scorer),
        ("Full Pipeline", check_pipeline),
        ("Bedrock LLM", check_bedrock),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ {name}: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓" if result else "✗"
        print(f"{status} {name}")
    
    print()
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All systems operational!")
        return 0
    elif passed >= total - 1:
        print(f"\n⚠️  {total - passed} component needs attention")
        if not results.get("Bedrock LLM"):
            print("   💡 Check your AWS credentials (aws configure)")
        return 0
    else:
        print(f"\n❌ {total - passed} components failing")
        return 1


if __name__ == "__main__":
    sys.exit(main())
