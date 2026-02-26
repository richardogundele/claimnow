# ClaimsNOW

**AI-Powered Motor Insurance Court Pack Analyser**

> Architecture, Build Guide & Portfolio Document

**Author:** Richard Ademola Ogundele  
**Version:** 1.0  
**Year:** 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Build Guide - Step by Step](#4-build-guide---step-by-step)
5. [MVP Scope vs Full Product](#5-mvp-scope-vs-full-product)
6. [How to Frame This on Your CV](#6-how-to-frame-this-on-your-cv)
7. [Spin-Off Projects Using the Same Pattern](#7-spin-off-projects-using-the-same-pattern)
8. [Suggested Build Timeline](#8-suggested-build-timeline)
9. [Contact](#9-contact)

---

## 1. Project Overview

ClaimsNOW is an end-to-end agentic AI system that ingests motor insurance court pack documents, extracts structured claims data using NLP, compares extracted hire rates against a reference database, scores each claim for potential inflation, and surfaces an explainable decision to the reviewer.

This project directly mirrors the technical scope of the **KTP Associate - Data Scientist** role at Whichrate. It demonstrates every core skill in that job description: document AI, NLP, ML pipeline design, MVP development, explainability, and change management thinking.

### 1.1 The Problem Being Solved

Motor insurers and solicitors spend hours manually reviewing court packs—legal document bundles containing credit hire invoices—to determine whether the daily rental rate claimed is fair or inflated.

Whichrate holds a database of **65 million+ UK vehicle hire rates**. The challenge is connecting that data intelligence to the document review process automatically, at scale.

### 1.2 What ClaimsNOW Delivers

| Capability | Description |
|------------|-------------|
| **Document Ingestion** | Accepts PDF court packs or invoices and extracts raw text using OCR and PDF parsing |
| **NLP Extraction** | Identifies hire dates, vehicle category, daily rate, total claimed, and hire company from unstructured text |
| **Rate Comparison** | Compares extracted rate against reference market rates for the same vehicle class, region, and hire period |
| **Claim Scoring** | Classifies each claim as Fair, Potentially Inflated, or Flagged for Review with a confidence score |
| **Explainability Layer** | Shows the reviewer exactly why a claim was flagged—which fields, which thresholds, which data |
| **Audit Trail** | Logs every decision with timestamps and reasoning for compliance and governance |

---

## 2. System Architecture

ClaimsNOW is built as a **five-layer pipeline**. Each layer is independently testable and deployable, which makes the project modular for portfolio demonstration and extensible for real-world production.

### 2.1 Architecture Layers

| Layer | Component | Responsibility |
|-------|-----------|----------------|
| **L1** | Document Ingestion | Accept PDF/image input, extract raw text via PyMuPDF or pdfplumber, handle scanned docs via Tesseract OCR |
| **L2** | NLP Extraction Agent | Parse extracted text using spaCy + regex patterns + Claude API to identify structured fields: hire dates, rates, vehicle type, claimant details |
| **L3** | Reference Matching Engine | Query the rate reference database (SQLite/CSV) to retrieve comparable market rates for the same vehicle class, region, and date range |
| **L4** | Scoring & Classification | Calculate inflation ratio, apply threshold rules, assign a Fair / Inflated / Flagged label with a numeric confidence score |
| **L5** | Explainability & Output | Generate a structured JSON report and human-readable summary explaining the decision, surfaced via a React or Streamlit dashboard |

### 2.2 Data Flow Diagram

```
PDF / Image Upload
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  [L1] Text Extraction  (PyMuPDF / Tesseract OCR)            │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  [L2] NLP Extraction   (spaCy + Claude API)  →  Structured  │
│                                                    JSON     │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  [L3] Rate Matching    (SQLite reference DB)  →  Market     │
│                                                    Rate     │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  [L4] Scoring          (Rules + ML model)     →  Label +    │
│                                                    Score    │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  [L5] Explainability   (Report + Dashboard)   →  Human      │
│                                                    Review   │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| PDF Parsing | PyMuPDF / pdfplumber | Extract text from native PDFs |
| OCR | Tesseract via pytesseract | Extract text from scanned/image PDFs |
| NLP Extraction | spaCy + custom regex | Named entity recognition, field extraction |
| AI Agent Layer | Claude API (claude-sonnet) | Intelligent extraction from complex documents |
| Reference Database | SQLite / CSV + pandas | Store and query market rate benchmarks |
| Scoring Logic | Python rules engine + sklearn | Inflation scoring and classification |
| Explainability | SHAP / custom reasoning layer | Explain why a claim was flagged |
| Backend API | FastAPI | Serve the pipeline as REST endpoints |
| Frontend Demo | React + Tailwind CSS | Interactive dashboard for claim review |
| Containerisation | Docker | Reproducible, deployable environment |
| Version Control | Git + GitHub | Portfolio hosting and code review |

---

## 4. Build Guide - Step by Step

This section walks through building ClaimsNOW from scratch. Each phase produces a working, demonstrable artifact. You do not need to complete all phases—**Phase 1 and 2 alone are enough for a strong interview demonstration**.

### Phase 1: Foundation (Day 1-2)

**Goal:** Upload a PDF, extract text, pull out key fields

1. Create project structure: `/claimsNow` with subfolders `/data`, `/src`, `/tests`, `/docs`, `/frontend`
2. Install dependencies:
   ```bash
   pip install pymupdf pdfplumber pytesseract spacy anthropic fastapi pandas
   ```
3. Build `document_loader.py` — accepts a PDF path, returns raw text using PyMuPDF, falls back to Tesseract for scanned pages
4. Build `extractor.py` — uses spaCy and regex to find:
   - Hire start date
   - Hire end date
   - Vehicle category (e.g., Group B, SUV)
   - Daily rate (GBP)
   - Total claimed amount
   - Hire company name
5. Write unit tests in `/tests` with 3 sample fake invoices to validate extraction accuracy
6. **Output:** Structured JSON
   ```json
   {
     "hire_days": 14,
     "daily_rate": 89.00,
     "vehicle_class": "Group C",
     "total": 1246.00
   }
   ```

### Phase 2: Rate Comparison Engine (Day 3-4)

**Goal:** Compare extracted rate against market benchmarks

7. Create a `reference_rates.csv` with 200+ synthetic market rate records:
   - Columns: `vehicle_class`, `region`, `hire_period_days`, `market_rate_low`, `market_rate_high`, `source_year`
8. Build `rate_matcher.py` — takes extracted JSON, queries the CSV/SQLite DB, returns the matching rate range for the same class, region, and period
9. Build `scorer.py` — calculates:
   - `inflation_ratio = claimed_rate / market_rate_high`
   - Applies thresholds:
     - Ratio < 1.1 = **Fair**
     - 1.1 - 1.4 = **Potentially Inflated**
     - \> 1.4 = **Flagged**
10. Add a `confidence_score` based on how many fields were successfully extracted (missing fields reduce confidence)

### Phase 3: AI Agent Enhancement (Day 5-6)

**Goal:** Use Claude API for intelligent extraction from complex documents

11. Create an `agent.py` that calls Claude API when the regex extractor fails to find all required fields
12. Design the prompt carefully: pass the raw document text, ask Claude to return only structured JSON with the required fields
13. Add a validation layer that checks Claude output against expected schema before proceeding
14. This makes ClaimsNOW an **agentic system**—it uses rule-based extraction first, escalates to AI when needed
15. Log all AI calls with input/output for auditability and governance demonstration

### Phase 4: FastAPI Backend (Day 7-8)

**Goal:** Serve the pipeline as a REST API

16. Create `main.py` with FastAPI app and a `POST /analyse` endpoint that accepts a PDF file upload
17. The endpoint calls: `document_loader` → `extractor` → `agent` (if needed) → `rate_matcher` → `scorer`
18. Returns a structured JSON response including:
    - Extracted fields
    - Market rate comparison
    - Claim verdict
    - Confidence score
    - Explanation text
19. Add a `GET /health` endpoint for deployment monitoring
20. Test with curl and Postman before connecting the frontend

### Phase 5: React Frontend Demo (Day 9-11)

**Goal:** Build a visual dashboard for interview demonstration

21. Create a React app with three screens:
    - **Upload Screen:** Drag and drop PDF
    - **Processing Screen:** Animated pipeline progress
    - **Results Screen:** Claim verdict with breakdown
22. Results screen shows:
    - Extracted fields table
    - Market rate comparison chart
    - Verdict badge (Fair / Inflated / Flagged)
    - Explanation panel
    - Confidence score meter
23. Connect to FastAPI backend via axios
24. Deploy frontend to GitHub Pages and backend to Railway or Render for a live demo link

---

## 5. MVP Scope vs Full Product

| Feature | Priority | Notes |
|---------|----------|-------|
| PDF text extraction | **Core** | Essential for all downstream processing |
| NLP field extraction | **Core** | The hardest part—proves AI engineering skill |
| Rate comparison logic | **Core** | The core business value of the system |
| Claim scoring + verdict | **Core** | The output that matters to the business |
| Explainability output | **Core** | Critical for trust and governance |
| Claude API agent fallback | Enhancement | Elevates to agentic AI—strong interview story |
| FastAPI REST backend | Enhancement | Shows production-readiness thinking |
| React frontend dashboard | Enhancement | Visual demo for non-technical interviewers |
| Docker containerisation | Enhancement | Shows DevOps and deployment awareness |
| ML classification model | Optional | Replace rules with a trained model for Phase 2 |
| Multi-document batch processing | Optional | Useful for scale but not needed for MVP |
| SHAP explainability integration | Optional | Advanced feature for model interpretability |

---

## 6. How to Frame This on Your CV

### 6.1 CV Project Entry

> **ClaimsNOW - AI-Powered Motor Insurance Court Pack Analyser**  
> *Personal Project | Python, Claude API, spaCy, FastAPI, React | 2026*
>
> - Architected a five-layer agentic AI pipeline that ingests PDF court pack documents, performs NLP-based extraction of hire rate data, and classifies insurance claims as fair or inflated against a 200+ record reference database
> - Implemented a hybrid extraction system combining rule-based spaCy NLP with Claude API fallback, enabling robust field extraction from both structured and unstructured legal documents
> - Designed an explainability layer that surfaces human-readable reasoning for every claim verdict, addressing governance and auditability requirements for professional services AI
> - Deployed via FastAPI backend and React frontend, producing a live demonstrable MVP with end-to-end document-to-verdict latency under 8 seconds

### 6.2 Interview Talking Points

Use these when asked about projects, technical experience, or the KTP role specifically:

**On the KTP role at Whichrate:**
> "I have already prototyped the core system you are looking to build. ClaimsNOW mirrors your technical requirements exactly—document ingestion, NLP extraction, rate comparison, and an explainable verdict. I built it to understand the problem space before this interview."

**On agentic AI:**
> "I designed ClaimsNOW as an agentic system rather than a fixed pipeline. The extractor first attempts deterministic NLP. When that fails—incomplete documents, unusual formatting—it escalates to a Claude API agent. That hybrid approach is how you build AI that works in the real world."

**On explainability:**
> "Every verdict in ClaimsNOW comes with a structured explanation—which fields were extracted, what the market comparison showed, and why the threshold was triggered. In professional services AI, explainability is not optional. It is the difference between a tool people trust and one they ignore."

**On change management:**
> "I designed the output layer with the human reviewer in mind. The dashboard does not replace the reviewer. It gives them everything they need to make a faster, more confident decision. That is the change management principle—AI that augments human judgement, not one that tries to replace it."

---

## 7. Spin-Off Projects Using the Same Pattern

The core pattern of ClaimsNOW—ingest document, extract structure, compare against reference, score, explain—applies to dozens of domains. Each project below can be built from the ClaimsNOW codebase with minimal changes.

| Project Name | What It Does | Target Role / Sector |
|--------------|--------------|----------------------|
| **ContractNOW** | Extracts clauses from contracts, flags risky terms vs standard templates, explains deviations | Legal tech, enterprise AI, consulting |
| **PolicyNOW** | Parses insurance policy documents, compares coverage terms across providers, highlights gaps | Insurtech, NHS, financial services |
| **GrantNOW** | Reads grant applications, scores them against funder criteria, surfaces ranked shortlist | Charity sector, GMCA, public sector AI |
| **AuditNOW** | Extracts line items from supplier invoices, compares against agreed contract rates, flags overcharges | NHS procurement, finance, enterprise |
| **EvidenceNOW** | Reads clinical evidence documents, extracts key findings, compares against treatment guidelines | NHS, Francis Crick Institute, health AI |
| **PlanningNOW** | Parses planning applications, checks against local policy criteria, produces compliance report | Local government, GMCA, urban tech |

> **Strategic note:** You do not need to build all of these. Build ClaimsNOW fully. Then note in interviews that the architecture is domain-agnostic and you have already mapped applications in healthcare, legal, and public sector. That shows systems thinking, not just coding ability.

---


## Contact

**Richard Ademola Ogundele**

- **Email:** Ogundelerichard27@gmail.com
- **LinkedIn:** [linkedin.com/in/richardogundele](https://linkedin.com/in/richardogundele)
- **Location:** Manchester, UK
- **Education:** MSc Artificial Intelligence (Distinction)
- **Professional Membership:** BCS Professional Member #995140023

---

### Quick Links

| Resource | Link |
|----------|------|
| **Live Demo** | *[Coming Soon]* |
| **Architecture Docs** | *[Coming Soon]* |

---

**Stack:** Python | spaCy | Claude API | FastAPI | React | SQLite  
**Features:** Document AI | NLP Extraction | Rate Comparison | Explainability
