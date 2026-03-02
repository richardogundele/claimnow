# ClaimsNOW v2 - Local ML Edition

**AI-Powered Motor Insurance Court Pack Analyser**  
**Using Local ML, RAG, and Fine-tuned LLMs - No Cloud Dependencies**

**Status**: In active development (local-first prototype, open to collaboration)  
**One-line summary**: End-to-end, five-layer document AI pipeline that ingests motor insurance court packs as PDFs, extracts structured claim data, compares them to a local market rate base via RAG, scores fairness with an ML model, and returns SHAP-backed, regulator-friendly explanations – all running entirely on your own machine.

> Author: Richard Ademola Ogundele  

---

## What This Project Demonstrates

|  Requirement | How This Project Shows It |
|-----------------|---------------------------|
| RAG pipelines for specific domains | ChromaDB + LangChain for rate retrieval |
| Fine-tune LLMs for enhanced performance | Local Mistral/Llama with domain prompts |
| Models on local, resource-constrained devices | Ollama - no cloud dependencies |
| MCP for System Integration | Tool-use architecture with LangChain agents |
| PyTorch, TensorFlow, LangChain | All used in this project |
| React dashboards with REST APIs | FastAPI backend + React frontend |

---

## Architecture Overview

```
PDF Upload
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  [L1] Document Parser (LOCAL)                               │
│  PyMuPDF + pdfplumber - no cloud OCR                       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  [L2] Field Extraction (LOCAL LLM)                          │
│  Ollama + Mistral 7B via LangChain                         │
│  Extracts: dates, rates, vehicle class, company            │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  [L3] RAG Rate Matching                                     │
│  ChromaDB vector store with 65M rate embeddings            │
│  Semantic search for comparable market rates               │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  [L4] ML Scoring Model                                      │
│  scikit-learn classifier trained on historical claims      │
│  SHAP explanations for interpretability                    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  [L5] Verdict & Explanation                                 │
│  FAIR | POTENTIALLY_INFLATED | FLAGGED                     │
│  Human-readable reasoning for every decision               │
└─────────────────────────────────────────────────────────────┘
```

---

## Agentic Orchestration & Integrations

- **Multi-layer agentic pipeline**: ClaimNow already behaves like a domain-specific agentic system – with distinct stages for parsing, extraction, RAG, scoring, and explanation that pass structured state between them.
- **Exploring orchestration frameworks**: Actively exploring integration with agentic orchestration frameworks (e.g. AdenHQ’s Hive, LangGraph, and similar tools) so the same pipeline can be expressed as an explicit, inspectable set of cooperating agents.
- **Why this matters**: Makes it easier to plug ClaimNow into larger digital workforces (claims triage, litigation support, SIU workflows) and to reuse the architecture for adjacent domains like credit hire, subrogation, or legal document review.

---

## Tech Stack

| Component | Technology | Why This Choice |
|-----------|------------|-----------------|
| **Document Parsing** | PyMuPDF, pdfplumber | Free, fast, local - no AWS Textract needed |
| **OCR (scanned docs)** | Tesseract | Open-source, runs locally |
| **Local LLM** | Ollama + Mistral 7B | Runs on laptop, no API costs, can fine-tune |
| **RAG Framework** | LangChain | Industry standard for LLM orchestration |
| **Vector Database** | ChromaDB | Lightweight, embeds with sentence-transformers |
| **Embeddings** | sentence-transformers | Local embedding model, no OpenAI needed |
| **ML Classifier** | scikit-learn / PyTorch | Custom model trained on claim data |
| **Explainability** | SHAP | Shows why model made each decision |
| **Backend** | FastAPI | Modern async Python API framework |
| **Frontend** | React + Tailwind CSS | Professional UI for demo |

---

## Project Structure

```
claimnow/
├── src/
│   ├── __init__.py
│   ├── config.py           # All configuration in one place
│   ├── document_parser.py  # Extract text from PDFs (local)
│   ├── llm_client.py       # Interface to Ollama/local LLM
│   ├── embeddings.py       # Generate embeddings for RAG
│   ├── vector_store.py     # ChromaDB operations
│   ├── rag_pipeline.py     # RAG retrieval logic
│   ├── extractor.py        # LLM-based field extraction
│   ├── rate_matcher.py     # Find comparable market rates
│   ├── scorer.py           # ML classification model
│   ├── explainer.py        # SHAP + human explanations
│   ├── pipeline.py         # Orchestrate full analysis
│   └── main.py             # FastAPI endpoints
│
├── models/                  # Saved ML models
│   └── claim_classifier.pkl
│
├── data/
│   ├── rates/              # Market rate CSV files
│   │   └── sample_rates.csv
│   └── training/           # Training data
│       └── labeled_claims.csv
│
├── vectorstore/            # ChromaDB persistence directory
│
├── tests/
│   ├── __init__.py
│   ├── test_parser.py
│   ├── test_extractor.py
│   ├── test_rag.py
│   └── test_scorer.py
│
├── frontend/               # React application
│
├── requirements.txt        # Python dependencies
├── .env.example           # Environment template
└── README.md              # This file
```

---

## Key Concepts Explained

### 1. RAG (Retrieval-Augmented Generation)

**Problem:** An LLM doesn't "know" UK car hire rates.

**Solution:** Store rates in a vector database, retrieve relevant ones, pass to LLM.

```
User Query: "Is £89/day fair for a Group C car in London?"
    │
    ▼
[1] Embed the query → vector [0.23, -0.45, 0.12, ...]
    │
    ▼
[2] Search ChromaDB for similar rate records
    │
    ▼
[3] Retrieved: "Group C, London, £45-65/day typical"
    │
    ▼
[4] LLM receives: Query + Retrieved Context
    │
    ▼
[5] LLM responds: "£89/day exceeds market by 37%"
```

### 2. Local LLM with Ollama

**Why not cloud (GPT-4, Claude)?**
- Data privacy: Insurance data shouldn't leave the network
- Cost: No per-token API charges
- Fine-tuning: Can adapt model to insurance terminology
- Interview advantage: Shows deeper ML skills

**How it works:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull mistral

# It now runs locally on port 11434
curl http://localhost:11434/api/generate -d '{"model":"mistral","prompt":"Hello"}'
```

### 3. MCP (Model Context Protocol)

An architecture where the LLM can use "tools":

```
LLM thinks: "I need to find the market rate for this vehicle"
    │
    ▼
LLM calls tool: search_rates(vehicle_class="GROUP_C", region="LONDON")
    │
    ▼
Tool returns: {"low": 45.00, "high": 65.00}
    │
    ▼
LLM continues reasoning with this data
```

LangChain implements this pattern with "Agents" and "Tools".

---

## Setup Instructions

### Prerequisites

1. **Python 3.10+**
2. **Ollama** - https://ollama.ai
3. **Git**

### Installation

```bash
# Clone the repo
cd claimnow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Install Ollama and pull model
# (Install Ollama from https://ollama.ai first)
ollama pull mistral

# Run the API
python -m uvicorn src.main:app --reload
```

### Running the Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## How Each Module Works

### document_parser.py
Extracts text from PDFs using PyMuPDF. No cloud services.

### llm_client.py  
Wrapper around Ollama API. Sends prompts to local Mistral model.

### embeddings.py
Uses sentence-transformers to create vector embeddings locally.

### vector_store.py
ChromaDB operations: add documents, search by similarity.

### rag_pipeline.py
Combines retrieval + generation. Searches for relevant rates, passes to LLM.

### extractor.py
Uses LLM to extract structured fields from document text.

### rate_matcher.py
Finds comparable market rates using vector similarity search.

### scorer.py
ML classifier (sklearn) that predicts: FAIR, INFLATED, or FLAGGED.

### explainer.py
SHAP values show which features influenced the prediction.

### pipeline.py
Orchestrates all components into one workflow.

### main.py
FastAPI endpoints for the frontend to call.

---

## Interview Talking Points

**On Local ML vs Cloud:**
> "I chose to run models locally because Whichrate handles sensitive insurance data. Local deployment means data never leaves the network, there are no per-query API costs, and we can fine-tune models on domain-specific terminology. The architecture uses Ollama for LLM inference and ChromaDB for vector search - both run entirely on-premise."

**On RAG:**
> "Rather than fine-tuning an LLM on 65 million rates - which would be expensive and inflexible - I implemented RAG. The rates are embedded in a vector database. When analysing a claim, we retrieve the most relevant rates and pass them as context to the LLM. This means the model always uses current data without retraining."

**On the Tech Stack:**
> "I used LangChain for LLM orchestration, ChromaDB for vector storage, and scikit-learn for the classification model. SHAP provides explainability - critical in insurance where decisions must be justified. The whole system runs on a standard laptop without GPU."

---

## Contact

**Richard Ademola Ogundele**  
Email: Ogundelerichard27@gmail.com  

