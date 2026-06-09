# MultiAgent-ResearchIQ

**ResearchIQ** is an autonomous AI Research & Competitive Intelligence Agent. Give it a topic or question and it produces a structured, cited intelligence report — via CLI, streaming API, or Streamlit UI.

## Features

| Capability | Description |
|------------|-------------|
| **Multi-agent planning** | Coordinator decomposes queries into 2–4 focused sub-tasks |
| **Parallel research** | Web Researcher (Tavily ReAct) + Doc Analyst (hybrid RAG) per sub-task |
| **Report synthesis** | Report Writer produces markdown with web/KB source sections |
| **Live agent trace** | SSE stream + Streamlit timeline (curated steps, no raw LLM tokens) |
| **Export** | Markdown + styled PDF to `data/reports/` |
| **Production API** | FastAPI with rate limiting, CORS for Streamlit, download endpoints |
| **Observability** | Local token/cost CLI + optional **LangSmith** tracing (no Langfuse) |

## Architecture

```
User query
    │
    ▼
Coordinator ──► sub_tasks
    │
    ├──► Web Researcher (×N) ──► Tavily + optional page fetch
    │
    └──► Doc Analyst (×N) ──► hybrid RAG (Qdrant + BM25 + rerank)
                │
                ▼
         Report Writer ──► markdown report ──► PDF export
```

Orchestration: **LangGraph** with parallel `Send()` dispatch per sub-task.

## Tech stack

| Layer | Technology |
|-------|------------|
| Orchestration | LangGraph |
| LLM | OpenAI `gpt-5.4-mini` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | Qdrant (local embedded or Docker) |
| Retrieval | BM25 + dense + RRF + cross-encoder rerank |
| Web search | Tavily (+ custom Tavily MCP server) |
| Backend | FastAPI, SSE streaming, in-memory rate limiting |
| Frontend | Streamlit |
| Export | Markdown + PDF (`fpdf2`) |
| Tracing | LangSmith (`LANGSMITH_*` env vars) |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ui]"

cp .env.example .env
# Required: OPENAI_API_KEY, TAVILY_API_KEY
# Optional: LANGSMITH_* for tracing (see below)
```

### Knowledge base (first run)

```bash
python scripts/download_seed_docs.py   # 10 curated ArXiv papers
python scripts/seed_knowledge_base.py  # index into Qdrant + BM25
```

## Quick start

**Terminal 1 — API**

```bash
uvicorn src.api.main:app --reload --port 8000
```

**Terminal 2 — Streamlit UI**

```bash
streamlit run ui/app.py
```

Open the UI, enter a research question, and watch the live agent trace update as SSE events arrive. The report and download buttons appear when the run completes.

**CLI (with export)**

```bash
python scripts/run_research.py "What are the latest advances in agentic RAG?"
# → data/reports/<timestamp>_<slug>.md and .pdf
```

**curl (streaming SSE)**

```bash
curl -N -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare transformer and Mamba architectures", "export_report": true}'
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/ingest` | Add PDF, URL, or text to the knowledge base |
| `POST` | `/research` | Run full pipeline; streams SSE events |
| `GET` | `/research/reports/{id}/markdown` | Download exported markdown |
| `GET` | `/research/reports/{id}/pdf` | Download exported PDF |

Rate limits (default, per IP): **3/min** on `/research`, **10/min** on `/ingest`. Set `RATE_LIMIT_ENABLED=false` in `.env` for unrestricted local dev.

## Environment variables

Copy `.env.example` → `.env`. Key settings:

```bash
# Required
OPENAI_API_KEY=
TAVILY_API_KEY=

# Qdrant (default: local, no Docker)
QDRANT_MODE=local

# Report export
REPORT_OUTPUT_DIR=data/reports
REPORT_EXPORT_ENABLED=true
RESEARCHIQ_API_URL=http://localhost:8000   # Streamlit only

# LangSmith tracing (optional)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=researchiq
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
# LANGSMITH_WORKSPACE_ID=...   # org-scoped API keys only

# Rate limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_RESEARCH_PER_MINUTE=3
RATE_LIMIT_INGEST_PER_MINUTE=10
```

Restart the API/CLI after changing `.env`. LangSmith traces appear under the `researchiq` project; the Streamlit UI shows a separate curated timeline (not raw LLM tokens).

## Project structure

```
src/
├── agents/           # Coordinator, web researcher, doc analyst, report writer
├── graph/            # LangGraph, SSE streaming, trace event queue
├── rag/              # Ingestion, hybrid search, index store
├── tools/            # Tavily search, scraper, RAG tools
├── mcp/              # Tavily MCP server
├── api/              # FastAPI routes + rate limiting
└── utils/            # Token cost, report export, LangSmith config
ui/
├── app.py            # Streamlit UI (incremental SSE consumer)
└── trace_renderer.py # Live timeline + metrics
data/reports/         # Exported reports (gitignored)
```

## Tests

```bash
pytest tests/ -v -m "not integration"
```

## Documentation

- [`docs/build_project_running.md`](docs/build_project_running.md) — Full build & run guide (Phases 1–4)
- [`AI_research_report_agent_plan.md`](AI_research_report_agent_plan.md) — Capstone project plan

## Roadmap

| Phase | Status | Scope |
|-------|--------|-------|
| 1 | Complete | Coordinator, LangGraph scaffold |
| 2 | Complete | RAG ingestion, hybrid search, `/ingest` |
| 3 | Complete | Web/doc agents, report writer, CLI |
| 4 | Complete | SSE API, Streamlit, export, live trace, rate limiting, LangSmith |
| Future | Planned | Auth/API keys, optional Notion export |

## Project status

Capstone project — AI Engineering Cohort. **v0.5.0**
