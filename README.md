# MultiAgent-ResearchIQ

**ResearchIQ** is an autonomous AI Research & Competitive Intelligence Agent that takes a topic or question and produces a structured intelligence report.

## What it does

- Accepts a natural-language research query
- Decomposes it into focused sub-tasks via a Coordinator agent
- Dispatches parallel Web Researcher and Document Analyst agents
- Synthesises findings into a cited, structured report saved to Notion

## Phase 2 Status (current)

- RAG ingestion pipeline (PDF, URL, text) with semantic chunking
- Dual indexing: Qdrant (dense) + BM25 (sparse, JSON-backed)
- Hybrid search with RRF fusion + cross-encoder re-ranking
- `POST /ingest` FastAPI endpoint
- Seed corpus: 10 curated ArXiv papers (transformers, LLMs, vision)

## Phase 1 Status (complete)

- Project scaffold with typed Pydantic settings
- Coordinator agent (`gpt-5.4-mini`) decomposes queries into 2–4 sub-tasks
- LangGraph orchestration with parallel `Send()` dispatch to stub workers
- Docker Compose for Qdrant (ready for Phase 2 RAG)

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (multi-agent, async) |
| LLM | OpenAI `gpt-5.4-mini` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | Qdrant (local embedded or Docker) |
| Retrieval | Hybrid BM25 + dense + RRF + cross-encoder re-ranking (Phase 2+) |
| MCP Servers | Custom Tavily MCP + Notion MCP (Phase 3+) |
| Backend | FastAPI + asyncio (Phase 4) |
| Frontend | Streamlit (Phase 4) |
| Observability | LangSmith + Langfuse |

## Documentation

- [`docs/ResearchIQ_System_Design.docx`](docs/ResearchIQ_System_Design.docx) — Full system & solution design document
- [`docs/build_project_running.md`](docs/build_project_running.md) — Build & run guide (Phase 1 & 2)
- [`AI_research_report_agent_plan.md`](AI_research_report_agent_plan.md) — Capstone project plan

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Edit .env and set OPENAI_API_KEY

# Qdrant — pick ONE option:
# Option A (default, no Docker): QDRANT_MODE=local in .env — nothing else to start
# Option B (Docker): install Docker Desktop, set QDRANT_MODE=server, then:
#   docker compose up -d
```

## Phase 2 — Knowledge Base Setup

**No Docker?** Use the default `QDRANT_MODE=local` in `.env`. Vectors are stored under `data/qdrant_storage/`.

```bash
# Download 10 ArXiv seed PDFs
python scripts/download_seed_docs.py

# Index into Qdrant + BM25 (requires OPENAI_API_KEY for embeddings)
python scripts/seed_knowledge_base.py

# Query the knowledge base (hybrid search)
python scripts/query_knowledge_base.py "self-attention mechanism in transformers"
python scripts/query_knowledge_base.py "vision transformer patch embeddings" --skip-rerank

# Start API server
uvicorn src.api.main:app --reload --port 8000

# Ingest additional documents
curl -X POST localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Transformer models use self-attention...", "title": "Notes"}'
```

## Usage

**Coordinator only** (structured research plan):

```bash
python scripts/run_coordinator.py "What are the latest advances in agentic RAG?"
```

**Full Phase 1 graph** (coordinator + parallel stubs + report writer stub):

```bash
python scripts/run_coordinator.py --graph "Competitive landscape for AI coding assistants in 2025"
```

## Tests

```bash
pytest tests/ -v -m "not integration"
```

## Project Structure

```
src/
├── config/settings.py      # Pydantic Settings
├── models/schemas.py       # ResearchPlan, AgentState models
├── agents/coordinator.py   # Query decomposition agent
├── graph/                  # LangGraph state, nodes, graph
├── rag/                    # Phase 2
├── tools/                  # Phase 2+
├── mcp/                    # Phase 3+
└── api/                    # Phase 4
```

## Roadmap

| Phase | Scope |
|-------|-------|
| **1** (complete) | Scaffold + coordinator agent |
| **2** (current) | RAG ingestion, hybrid search, re-ranking, `/ingest` |
| **3** | Web/doc agents, Tavily MCP |
| **4** | Report writer, Notion MCP, FastAPI `/research`, Streamlit |

## Project Status

In active development — Capstone Project, AI Engineering Cohort
