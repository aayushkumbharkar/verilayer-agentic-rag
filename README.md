# VeriLayer 🛡️

> **Trust Layer for LLM Systems** — Production-grade Agentic RAG with Self-Correction, Hybrid Retrieval, and Claim Verification.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Langfuse](https://img.shields.io/badge/Langfuse-observability-purple.svg)](https://langfuse.com)
[![Docker](https://img.shields.io/badge/Docker-compose-blue.svg)](https://www.docker.com/)

VeriLayer is an enterprise-grade agentic RAG (Retrieval-Augmented Generation) system built to solve hallucination and factuality issues in LLM applications. By structuring the query-to-generation pipeline as a stateful, cyclic computational graph, VeriLayer guarantees that every assertion returned to the user is verified, scored, and cross-referenced with concrete source documents.

---

## System Architecture & Workflow

VeriLayer operates as a stateful multi-agent system powered by **LangGraph**. The workflow progresses through cyclic verification loop phases:

```
[Query Input] ──> [Planner] ──> [Decomposed Queries] ──> [Hybrid Search]
                                                               │
[Output Answer] <── [Verification Check] <── [Scorer] <── [Verifier] <── [Document Grader]
       │                   │
       │ (Hallucinated)    │ (Supported)
       └──> [Rewriter] ────┘
```

1. **Query Planning & Deconstruction**: Raw queries are analyzed and broken down into independent sub-queries.
2. **Parallel Hybrid Retrieval**: Sub-queries execute lexical (BM25) and semantic (Dense Vector) searches in parallel across OpenSearch clusters.
3. **Source Grading**: Retrieved documents are evaluated for relevance; non-relevant documents are culled to prevent context pollution.
4. **Answer Generation**: An LLM generates a comprehensive response grounded in the graded context.
5. **Atomic Claim Extraction**: The generated response is parsed into distinct, testable factual assertions.
6. **Cross-Reference Verification**: Each claim is mathematically checked against the exact source text of retrieved documents.
7. **Confidence Scoring**: A trust/groundedness metric is calculated.
8. **Self-Correction & Rewrite Loop**: If any claims are flagged as hallucinated or ungrounded, the system initiates a rewriting phase (up to 2 retries) to fix assertions before final delivery.

---

## Codebase Map: Where We Use and What For

Below is the mapping of components, directories, and technologies used throughout VeriLayer, along with their engineering purpose:

| Technology / Module | Location | What it is used for | Why we use it |
| :--- | :--- | :--- | :--- |
| **LangGraph (Graph Engine)** | [`src/agents/graph/verilayer_graph.py`](file:///c:/verilayer-agentic-rag/src/agents/graph/verilayer_graph.py) | Defines the state transitions, conditional routing, and nodes representing the agentic pipeline. | Allows stateful, multi-step agent coordination and handles retry/self-correction loops cleanly. |
| **Planner Node** | [`src/agents/nodes/planner.py`](file:///c:/verilayer-agentic-rag/src/agents/nodes/planner.py) | Decomposes complex user inputs into multiple target search criteria. | Boosts retrieval recall on multi-faceted questions. |
| **Sub-Query Retriever** | [`src/agents/nodes/sub_query_retriever.py`](file:///c:/verilayer-agentic-rag/src/agents/nodes/sub_query_retriever.py) | Coordinates executing parallel search requests. | Gathers distinct data points across different query angles. |
| **Document Grader** | [`src/agents/nodes/document_grader.py`](file:///c:/verilayer-agentic-rag/src/agents/nodes/document_grader.py) | Filters out low-relevance retrieval results. | Prevents LLM context window cluttering and "Lost in the Middle" syndrome. |
| **Generator Node** | [`src/agents/nodes/generator.py`](file:///c:/verilayer-agentic-rag/src/agents/nodes/generator.py) | Synthesizes response drafts from relevant chunks. | Prompts LLMs to construct highly coherent and grounded paragraphs. |
| **Claim Extractor** | [`src/agents/nodes/claim_extractor.py`](file:///c:/verilayer-agentic-rag/src/agents/nodes/claim_extractor.py) | Isolates individual factual claims from raw paragraphs. | Simplifies verification by breaking down a long answer into discrete statements. |
| **Claim Verifier** | [`src/agents/nodes/claim_verifier.py`](file:///c:/verilayer-agentic-rag/src/agents/nodes/claim_verifier.py) | Performs NLI (Natural Language Inference) checks of claims against retrieved chunks. | Identifies exact sentences that back up or contradict the LLM's assertions. |
| **Confidence Scorer** | [`src/agents/nodes/confidence_scorer.py`](file:///c:/verilayer-agentic-rag/src/agents/nodes/confidence_scorer.py) | Aggregates verification outputs into statistical trust scores. | Provides transparency by returning a definitive confidence level per response. |
| **Rewriter Node** | [`src/agents/nodes/rewriter.py`](file:///c:/verilayer-agentic-rag/src/agents/nodes/rewriter.py) | Automatically edits hallucinated assertions using source truth. | Implements a self-correcting RAG loop to avoid delivering incorrect facts. |
| **OpenSearch 2.11** | [`src/retrieval/`](file:///c:/verilayer-agentic-rag/src/retrieval/) | Implements Hybrid BM25 + Dense Vector Search. | Provides unified keyword matching and semantic vector retrieval with high performance. |
| **Jina Embeddings** | [`src/retrieval/embeddings.py`](file:///c:/verilayer-agentic-rag/src/retrieval/embeddings.py) | Generates high-dimension (1024-d) dense vector embeddings. | Captures multi-lingual semantic context and long-sequence alignment. |
| **PostgreSQL 15** | [`src/ingestion/postgres_writer.py`](file:///c:/verilayer-agentic-rag/src/ingestion/postgres_writer.py) | Serves as the primary metadata store and source document catalog. | Ensures relational integrity for document hierarchies, chunks, and citations. |
| **Redis 7** | [`src/observability/redis_cache.py`](file:///c:/verilayer-agentic-rag/src/observability/redis_cache.py) | Caches pipeline traces and generated outputs. | Lowers query latency and reduces operational token costs for identical sub-queries. |
| **Langfuse** | [`src/observability/langfuse_client.py`](file:///c:/verilayer-agentic-rag/src/observability/langfuse_client.py) | Traces graph execution, token usage, latency, and costs. | Provides observability and audit trails for nested LLM chains. |
| **Gradio 6** | [`ui/`](file:///c:/verilayer-agentic-rag/ui/) | Hosts the user-facing web app interface. | Provides an interactive dashboard for users to upload files, query the RAG, and view traces. |

---

## Technical Features

* **Multi-Stage Verification Loop**: If a claim is marked as unsupported (`unsupported` or `partial`), the `rewriter` node edits the assertion using the source context and feeds the draft back into the verifier (up to `MAX_RETRIES`).
* **Hybrid Lexical & Semantic Retrieval**: Merges keyword-based BM25 indexes and Dense Vector representation under Jina AI embeddings, employing reciprocal rank fusion (RRF) for top-k selection.
* **Granular Observability**: Integrated tracing exports structured spans to Langfuse, capturing LLM token costs, graph routing decisions, execution latencies, and retrieval scores.
* **Robust Chunking and Extraction**: The ingestion pipeline utilizes a recursive character text splitter accompanied by metadata extraction (sections, clauses, and titles) for precise back-referencing.

---

## Local Development & Setup

### Prerequisites
* Docker and Docker Compose
* Python 3.11+
* Git

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/aayushkumbharkar/verilayer-agentic-rag.git
   cd verilayer-agentic-rag
   ```

2. Copy and configure the environment variables:
   ```bash
   cp .env.example .env
   # Add your Groq, Jina, and Langfuse credentials to .env
   ```

3. Spin up the infrastructure and services:
   ```bash
   docker compose up -d --build
   ```

4. Verify service status and health:
   ```bash
   curl http://localhost:8000/health/services
   ```
   *Expected response:*
   ```json
   {
     "status": "healthy",
     "services": {
       "postgresql": {"status": "healthy"},
       "opensearch": {"status": "healthy"},
       "redis": {"status": "healthy"}
     }
   }
   ```

---

## API Documentation

VeriLayer exposes FastAPI endpoints (interactive docs available at `http://localhost:8000/docs`):

* `POST /ingest` — Ingest document content into PostgreSQL and OpenSearch indexes.
* `POST /ingest/pdf` — Parse and index document PDF files.
* `POST /search/hybrid` — Perform RRF-based hybrid search on queries.
* `POST /verify` — Trigger the stateful LangGraph agentic loop (generation, claim verification, and correction).
* `GET /metrics` — Retrieve current system accuracy, hallucination rates, and execution time statistics.

---

## Testing & Quality Assurance

VeriLayer is covered by a test suite testing both standalone components and multi-agent flows:

### Run Tests
```bash
# Set up a virtual environment and dependencies
pip install -r requirements.txt

# Run unit and integration tests
pytest tests/ -v
```

* **Unit Tests**: [`tests/unit/`](file:///c:/verilayer-agentic-rag/tests/unit/) (validates schema integrity, OpenSearch indexing, BM25 performance, and LangGraph state schemas).
* **Integration Tests**: [`tests/integration/`](file:///c:/verilayer-agentic-rag/tests/integration/) (evaluates the complete retrieval-to-rewrite flow and ingestion pipeline).
