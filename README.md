# MultiAgent-ResearchIQ

**ResearchIQ** is an autonomous AI Research & Competitive Intelligence Agent that takes a topic or question and produces a structured intelligence report.

## What it does

- Accepts a natural-language research query
- Decomposes it into focused sub-tasks via a Coordinator agent
- Dispatches parallel Web Researcher and Document Analyst agents
- Synthesises findings into a cited markdown report with **PDF export**

## Phase 4 Status (current — MVP)

- `POST /research` — SSE streaming research pipeline
- Markdown + PDF export to `data/reports/`
- Report download endpoints (`/research/reports/{id}/markdown|pdf`)
- Streamlit UI with live progress + download buttons

## Phase 3 Status (complete)

- Web Researcher agent (Tavily search + page fetch via ReAct)
- Document Analyst agent (hybrid RAG over seeded knowledge base)
- Report Writer agent (structured markdown synthesis)
- LangGraph dual `Send()` dispatch: web + doc agents per sub-task in parallel
- Tavily MCP server (`src/mcp/tavily_server.py`)
- End-to-end CLI: `scripts/run_research.py`

## Phase 2 Status (complete)

- RAG ingestion pipeline (PDF, URL, text) with semantic chunking
- Dual indexing: Qdrant (dense) + BM25 (sparse, JSON-backed)
- Hybrid search with RRF fusion + cross-encoder re-ranking
- `POST /ingest` FastAPI endpoint
- Seed corpus: 10 curated ArXiv papers (transformers, LLMs, vision)

## Phase 1 Status (complete)

- Project scaffold with typed Pydantic settings
- Coordinator agent (`gpt-5.4-mini`) decomposes queries into 2–4 sub-tasks
- LangGraph orchestration with parallel `Send()` dispatch to web + doc agents

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (multi-agent, async) |
| LLM | OpenAI `gpt-5.4-mini` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | Qdrant (local embedded or Docker) |
| Retrieval | Hybrid BM25 + dense + RRF + cross-encoder re-ranking |
| MCP Servers | Custom Tavily MCP |
| Backend | FastAPI + SSE streaming |
| Frontend | Streamlit |
| Export | Markdown + PDF (`fpdf2`) |

## Documentation

- [`docs/build_project_running.md`](docs/build_project_running.md) — Build & run guide (Phases 1–4)
- [`AI_research_report_agent_plan.md`](AI_research_report_agent_plan.md) — Capstone project plan

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ui]"

cp .env.example .env
# Edit .env — set OPENAI_API_KEY, TAVILY_API_KEY
```

## Quick start (Phase 4)

**Terminal 1 — API:**

```bash
uvicorn src.api.main:app --reload --port 8000
```

**Terminal 2 — Streamlit UI:**

```bash
streamlit run ui/app.py
```

**Or CLI with export:**

```bash
python scripts/run_research.py "What are the latest advances in agentic RAG?"
# Writes data/reports/<timestamp>_<slug>.md and .pdf
```

**Or curl (streaming):**

```bash
curl -N -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare transformer and Mamba architectures", "export_report": true}'
```

## Phase 2 — Knowledge Base Setup

```bash
python scripts/download_seed_docs.py
python scripts/seed_knowledge_base.py
python scripts/query_knowledge_base.py "self-attention mechanism in transformers"
```

## Tests

```bash
pytest tests/ -v -m "not integration"
```

## Project Structure

```
src/
├── agents/          # Coordinator, web, doc, report writer
├── graph/           # LangGraph + streaming.py
├── rag/             # Ingestion + hybrid search
├── tools/           # Search, scraper, RAG tools
├── mcp/             # Tavily MCP server
├── api/             # FastAPI (/ingest, /research)
└── utils/           # Token cost, report export
ui/
└── app.py           # Streamlit UI
data/reports/        # Exported markdown + PDF (gitignored)
```

## Roadmap

| Phase | Scope |
|-------|-------|
| **1–3** (complete) | Planner, RAG, agents, CLI |
| **4** (current MVP) | Streaming API, markdown/PDF export, Streamlit |
| **Future** | Langfuse, rate limiting, optional Notion export |

## Project Status

In active development — Capstone Project, AI Engineering Cohort
