# VeriLayer 🛡️

> **Trust Layer for LLM Systems** — Production-grade Agentic RAG with Hybrid Retrieval and Claim Verification

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Langfuse](https://img.shields.io/badge/Langfuse-observability-purple.svg)](https://langfuse.com)

---

## What VeriLayer Does

1. **Plans** the query before retrieval
2. **Decomposes** into sub-queries
3. **Retrieves** via hybrid search (BM25 + Jina embeddings)
4. **Generates** an answer with Groq LLM
5. **Extracts** atomic claims from the answer
6. **Verifies** each claim against retrieved sources
7. **Scores** confidence per claim
8. **Rewrites** hallucinated claims (up to 2 retries)
9. **Returns** grounded answer with citations, confidence, and full agent trace

Every step is traced in **Langfuse** with model names, token usage, and span hierarchy.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI |
| Database | PostgreSQL 15 |
| Search | OpenSearch 2.11 (BM25 + KNN) |
| Cache | Redis 7 |
| LLM | Groq (llama-3.3-70b-versatile) |
| Embeddings | Jina v3 |
| Orchestration | LangGraph |
| Observability | Langfuse |
| UI | Gradio |
| Infra | Docker Compose |

---

## Quick Start

### Prerequisites
- Docker + Docker Compose
- Python 3.11+

### 1. Clone and configure
```bash
git clone https://github.com/aayushkumbharkar/verilayer-agentic-rag.git
cd verilayer-agentic-rag
cp .env.example .env
# Edit .env and add your API keys
```

### 2. Start all services
```bash
docker compose up -d
```

### 3. Verify services are healthy
```bash
curl http://localhost:8000/health
# {"status":"ok","service":"VeriLayer","version":"0.1.0"}

curl http://localhost:8000/health/services
# {"status":"healthy","services":{"postgresql":{"status":"healthy"},...}}
```

### 4. Open API docs
```
http://localhost:8000/docs
```

---

## Project Structure

```
verilayer-agentic-rag/
├── src/
│   ├── api/             # FastAPI routes
│   ├── agents/          # LangGraph nodes + graph
│   ├── core/            # Config + logging
│   ├── evaluation/      # Metrics computation
│   ├── ingestion/       # PDF/text ingestion pipeline
│   ├── models/          # Canonical Pydantic schemas
│   ├── observability/   # Langfuse + Redis + audit logger
│   ├── rag/             # Generator + prompt templates
│   └── retrieval/       # BM25, embeddings, hybrid search
├── ui/                  # Gradio app
├── tests/               # Unit + integration tests
├── docker/              # Dockerfiles
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness probe |
| GET | `/health/services` | Readiness probe (PG + OS + Redis) |
| POST | `/ingest` | Ingest a document (Phase 2) |
| POST | `/search/bm25` | BM25 search (Phase 3) |
| POST | `/search/hybrid` | Hybrid search (Phase 4) |
| POST | `/rag/query` | Basic RAG (Phase 5) |
| POST | `/verify` | Full agentic verification (Phase 6) |
| GET | `/metrics` | Evaluation metrics (Phase 7) |

---

## /verify Response Contract

```json
{
  "query": "string",
  "final_answer": "string",
  "confidence": 0.87,
  "status": "verified | partial | unsafe",
  "claims": [
    {
      "text": "string",
      "verdict": "supported | unsupported | partial",
      "confidence": 0.91,
      "sources": [{"document_id": "...", "chunk_id": "...", "text": "..."}]
    }
  ],
  "trace": [
    {"step": "planner", "details": "...", "latency_ms": 120}
  ],
  "metadata": {"retrieval_docs": 5, "retries": 1, "latency_total_ms": 1240}
}
```

---

## Observability

All agent steps are traced in [Langfuse](https://cloud.langfuse.com) with:
- Descriptive trace names per query
- Model name + token usage on every LLM call
- Full span hierarchy: trace → retriever → verifier → rewriter
- Confidence scores logged as Langfuse scores

---

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

---

## Build Phases

- [x] Phase 1: Infrastructure + Docker + Langfuse tracing
- [x] Phase 2: Ingestion pipeline
- [x] Phase 3: BM25 search
- [x] Phase 4: Hybrid search
- [x] Phase 5: Basic RAG
- [x] Phase 6: Agentic system (LangGraph)
- [x] Phase 7: Observability + evaluation
- [x] Phase 8: Gradio UI
