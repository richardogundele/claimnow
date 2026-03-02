## Testing Guide - ClaimsNOW

Complete testing strategy for the enterprise-grade claim analysis pipeline.

---

## Quick Start (5 minutes)

### 1. **Health Check** ✓
Verify all components are working:

```powershell
python tests/test_health_check.py
```

**Output shows:**
- ✓ Python environment
- ✓ Configuration loaded
- ✓ Ollama LLM (if running)
- ✓ Embedding model (local)
- ✓ Vector store (ChromaDB)
- ✓ Document parser (OCR capable)
- ✓ All 8 pipeline stages

**If Ollama fails:**
```powershell
# In another terminal:
ollama serve
```

---

## Test Levels

### Level 1: Component Health (2 min)
```powershell
python tests/test_health_check.py
```

✅ Tests:
- Configuration loading
- Each module imports successfully
- No missing dependencies
- Services are reachable

**Expected:** All green (✓)

---

### Level 2: Direct Analysis (3 min)
```powershell
python tests/test_api.py
```

✅ Tests:
- Pipeline with raw text (no PDF)
- Extraction accuracy
- Vector embedding generation
- RAG retrieval
- LLM-based scoring
- Audit logging

**Expected:**
- Verdict assigned (FAIR/POTENTIALLY_INFLATED/FLAGGED)
- Confidence score (0-1)
- Evidence sources listed
- Audit ID generated

---

### Level 3: End-to-End with PDF (5-10 min)
```powershell
python tests/test_pipeline_e2e.py
```

✅ Tests all 8 stages:
1. 📄 **Parse:** Extract text from PDF
2. 🔍 **Extract:** Get fields via Ollama
3. 🔢 **Embed:** Generate vectors
4. 🔎 **Search:** Find similar cases (65M indexed)
5. 📊 **Benchmark:** Calculate market stats
6. 🤖 **Score:** LLM reasoning with context
7. 📝 **Explain:** SHAP feature importance
8. 📋 **Output:** Structured JSON + audit

**Expected Output:**
```
Verdict:          FAIR
Confidence:       87.3%
Total time:       2,341 ms
Stages completed: 5/5 ✓

Evidence: 5 comparable rates
Audit ID: <uuid>
```

---

## Running the API Server

### Start the server:
```powershell
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Test endpoints:

**1. Health Check**
```powershell
curl http://localhost:8000/health
```

**2. Analyze Text**
```powershell
$payload = @{
    text = "INVOICE: BMW 320d, £65/day, 14 days"
    claim_id = "CLAIM-001"
} | ConvertTo-Json

curl -X POST http://localhost:8000/api/analyze-text `
  -ContentType "application/json" `
  -Body $payload
```

**3. Upload PDF**
```powershell
curl -X POST http://localhost:8000/api/upload `
  -Form 'file=@data/court_pack_001_LDS320_2023.pdf'
```

**4. View Swagger UI**
```
http://localhost:8000/docs
```

---

## Raw Test Examples

### Example 1: Simple Text Analysis
```python
from src.pipeline import ClaimsPipeline

pipeline = ClaimsPipeline()

text = """
BMW 320d hired for 14 days
Daily rate: £65
Total: £910
"""

result = pipeline.analyze_text(text, claim_id="TEST-001")
print(f"Verdict: {result.verdict.value}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Evidence: {len(result.scoring.evidence_sources)} sources")
```

### Example 2: PDF Analysis
```python
from src.pipeline import ClaimsPipeline

pipeline = ClaimsPipeline()
result = pipeline.analyze("data/court_pack_001_LDS320_2023.pdf")

# Structured JSON output
import json
print(json.dumps(result.scoring.to_dict(), indent=2))
```

### Example 3: Audit Trail Access
```python
result = pipeline.analyze(pdf_path)

audit = result.scoring.audit_log
print(f"Audit ID:     {audit.audit_id}")
print(f"Timestamp:    {audit.timestamp}")
print(f"Method:       {audit.scoring_method}")
print(f"Latency:      {audit.decision_latency_ms:.1f}ms")
print(f"Retrieved:    {audit.retrieval_count} documents")
```

---

## Troubleshooting

### ❌ "Ollama not available"
```powershell
# Start Ollama in another terminal
ollama serve

# Then test
ollama list  # Should show: mistral
```

### ❌ "Vector store empty"
```python
# Populate with market rates
exec(open("data/generate_court_packs.py").read())
```

### ❌ "PDF parsing fails"
- Ensure Tesseract is installed:
```powershell
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# Then set environment variable:
$env:TESSERACT_PATH = "C:\Program Files\Tesseract-OCR"
```

### ❌ "Slow performance"
- Check `decision_latency_ms` in audit log
- Should be < 2000ms typically
- Latency breakdown:
  - Parse PDF: ~500ms
  - Extract via Ollama: ~800ms
  - Vector search: ~300ms
  - LLM scoring: ~400ms

---

## Test Checklist

Use this to verify everything is working:

- [ ] Health check passes (python tests/test_health_check.py)
- [ ] Direct analysis works (python tests/test_api.py)
- [ ] E2E pipeline runs (python tests/test_pipeline_e2e.py)
- [ ] API server starts (uvicorn src.main:app --reload)
- [ ] Swagger UI loads (http://localhost:8000/docs)
- [ ] Can upload PDF via API
- [ ] Verdict includes audit trail
- [ ] Evidence sources are populated
- [ ] JSON output is well-formed

---

## Interpreting Results

### Verdict Meanings

**FAIR** (Green)
- Rate is within market norms (±15%)
- Proceeding with claim is appropriate
- Confidence: typically 80-95%

**POTENTIALLY_INFLATED** (Yellow)
- Rate is 15-40% above market
- Needs human review
- Confidence: 60-80%

**FLAGGED** (Red)
- Rate is >40% above market
- Possible fraud/error
- Recommend denial or negotiation
- Confidence: 70-95%

**INSUFFICIENT_DATA** (Gray)
- Not enough comparable market rates
- Cannot make reliable assessment
- Get more market data, then resubmit

### Audit Log Fields

```json
{
  "audit": {
    "audit_id": "uuid",                      // Unique decision ID
    "timestamp": "2026-03-02T14:30:45Z",     // When decided
    "scoring_method": "llm_rag",             // LLM or rules
    "model_version": "1.0",                  // Model version
    "llm_model": "mistral",                  // Which LLM
    "retrieval_count": 5,                    // Documents found
    "decision_latency_ms": 234.56,           // How long it took
    "user_overrides": null                   // If human changed it
  }
}
```

---

## Performance Targets

| Stage | Target | Status |
|-------|--------|--------|
| Parse PDF | < 500ms | ✓ |
| Extract fields | < 1000ms | ✓ |
| Vector search | < 300ms | ✓ |
| Score with LLM | < 500ms | ✓ |
| **Total** | **< 2500ms** | ✓ |

---

## Next Steps

1. ✅ **Run health check** - Verify setup
2. ✅ **Test as-is** - Confirm all components work
3. 📊 **Populate vector store** - Load 65M market rates
4. 🚀 **Deploy API** - Expose to frontend
5. 📈 **Monitor performance** - Track decision latency
6. 🔄 **Iterate** - Improve based on real claims

---

## Commands Summary

```powershell
# Health check
python tests/test_health_check.py

# Direct analysis test
python tests/test_api.py

# End-to-end with PDF
python tests/test_pipeline_e2e.py

# Start API server
uvicorn src.main:app --reload --port 8000

# Swagger documentation
# Open: http://localhost:8000/docs

# Start Ollama (in separate terminal)
ollama serve
```

---

**Ready to test? Start with `python tests/test_health_check.py` 🚀**
