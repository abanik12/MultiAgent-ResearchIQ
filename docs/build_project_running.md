# ResearchIQ — Build & Run Guide (Phase 1, 2 & 3)

This document explains **what was built**, **why**, and **how to run it** — written for someone new to the codebase who wants to understand what happens step-by-step when code runs.

---

## Table of contents

1. [What ResearchIQ is](#what-researchiq-is)
2. [Project setup, repo & key decisions](#project-setup-repo--key-decisions)
3. [Build chronology (last few days)](#build-chronology-last-few-days)
4. [Project layout](#project-layout)
5. [Phase 1 — Coordinator & LangGraph scaffold](#phase-1--coordinator--langgraph-scaffold)
6. [Phase 2 — RAG knowledge base](#phase-2--rag-knowledge-base)
7. [Phase 3 — Specialist agents & full pipeline](#phase-3--specialist-agents--full-pipeline)
8. [API testing walkthrough](#api-testing-walkthrough)
9. [How Phase 1, 2, and 3 connect](#how-phase-1-and-phase-2-connect)
10. [Environment & configuration](#environment--configuration)
11. [Commands cheat sheet](#commands-cheat-sheet)
12. [What comes next (Phase 4+)](#what-comes-next-phase-3)
13. [Troubleshooting](#troubleshooting)
14. [Capstone talking points](#capstone-talking-points)

---

## What ResearchIQ is

ResearchIQ is an autonomous research platform. The end goal:

> User asks a research question → system produces a structured intelligence report.

We are building in phases. **Today (Phase 1 + 2 + 3):**

| Phase | What it does | Status |
|-------|--------------|--------|
| **Phase 1** | Breaks a big question into 2–4 focused sub-tasks (coordinator agent) | Complete |
| **Phase 2** | Indexes curated documents and retrieves relevant chunks (RAG) | Complete |
| **Phase 3** | Web + doc agents search in parallel; report writer synthesizes markdown | Complete |
| **Phase 4** | Full API, Streamlit UI, Notion export | Not started |

Think of Phase 1 as the **research manager** (plans only). Phase 2 is the **library** (stores and finds information). Phase 3 connects them with live web search and synthesis.

**Related docs:**
- Capstone master plan: [`AI_research_report_agent_plan.md`](../AI_research_report_agent_plan.md)
- System design: [`docs/ResearchIQ_System_Design.docx`](ResearchIQ_System_Design.docx) (if present)

---

## Project setup, repo & key decisions

### Where the project lives

| Item | Value |
|------|-------|
| Local path | `/Users/relanto/CLAUDE WORK/capstone project/researchiq/` |
| GitHub remote | https://github.com/abanik12/MultiAgent-ResearchIQ.git |
| Default branch | `main` |

The repo on GitHub already contained the capstone plan and system design docs before Phase 1 code was merged.

### Model choices (decided during Phase 1 planning)

| Setting | Value | Why |
|---------|-------|-----|
| LLM | `gpt-5.4-mini` | OpenAI-only for capstone; you had `OPENAI_API_KEY` ready |
| Embeddings | `text-embedding-3-small` | Fast, cheap, good quality for dense retrieval |

Originally the Claude-generated capstone plan listed Claude Sonnet — we switched to OpenAI for your setup.

### Phase 2 RAG strategy: pre-index vs query-time

We explicitly chose **not** to build the vector index from the user's query at runtime.

| Approach | Used? | Reason |
|----------|-------|--------|
| **Pre-index curated docs** (seed + `/ingest`) | Yes | Reliable demo, tests hybrid RAG, matches capstone design |
| **Build index from query at runtime** | No | Slow, expensive, blurs web vs doc agent roles |

Flow:
- **Ingest time** — download PDFs, chunk, embed, store (once or via `/ingest`)
- **Query time** — `hybrid_search()` reads existing index only

Live web search is reserved for the **Web Researcher** agent in Phase 3 (Tavily), not the vector DB.

### Docker vs local Qdrant

During Phase 2 setup you hit `command not found: docker`. Docker is **optional**.

| Mode | `.env` | When to use |
|------|--------|-------------|
| **Local (default)** | `QDRANT_MODE=local` | No Docker; vectors in `data/qdrant_storage/` |
| **Server** | `QDRANT_MODE=server` + `docker compose up -d` | Production-like / shared concurrent access |

Local mode uses Qdrant's embedded file storage via `qdrant-client` — same API, no container.

### OpenAI API key — where it comes from

If the coordinator works but you never exported a key in the terminal, it is almost certainly in **`.env`** (gitignored):

```bash
grep OPENAI_API_KEY .env
```

`scripts/run_coordinator.py` calls `load_dotenv()` which loads `.env` automatically. The key is **not** in GitHub.

---

## Build chronology (last few days)

Rough order of what was built and debugged in this project:

| Step | What happened |
|------|---------------|
| 1 | Read capstone plan (`AI_research_report_agent_plan.md`); scoped **Phase 1** (scaffold + coordinator) |
| 2 | Created project structure, Pydantic settings, coordinator with structured output |
| 3 | LangGraph graph with `Send()` parallel dispatch + stub workers |
| 4 | Pushed to GitHub; fixed git issues (orphan commit → rebase onto remote `main`) |
| 5 | Added **token + cost summary** to coordinator CLI (`src/utils/token_cost.py`) |
| 6 | Scoped and implemented **Phase 2** RAG (ingestion, dual index, hybrid search, `/ingest`) |
| 7 | Curated **10 ArXiv papers** (transformers, LLMs, vision, Mamba, GPT-4) |
| 8 | You successfully ran `download_seed_docs.py` + `seed_knowledge_base.py` (~**1,380 chunks**) |
| 9 | Fixed `docker` not found → default **local Qdrant** |
| 10 | Fixed hybrid search (`QdrantClient.search` → `query_points` API change) |
| 11 | Added `query_knowledge_base.py` (heredoc + `load_dotenv()` was causing `AssertionError`) |
| 12 | FastAPI running; `/health` returns `ok`; API-level `/ingest` testing documented |

---

## Project layout

```
researchiq/
├── src/
│   ├── config/settings.py       # All env vars (API keys, models, paths)
│   ├── models/schemas.py        # Pydantic data shapes (ResearchPlan, DocumentChunk, etc.)
│   ├── agents/coordinator.py    # Phase 1: query → sub-tasks
│   ├── graph/                   # Phase 1: LangGraph orchestration
│   │   ├── state.py             # Shared state between nodes
│   │   ├── nodes.py             # Node functions (coordinator, stubs)
│   │   └── graph.py             # Wires nodes together
│   ├── rag/                     # Phase 2: ingestion + retrieval
│   │   ├── index_store.py       # Qdrant + BM25 dual index
│   │   ├── ingestion.py         # Load PDF/URL/text → chunk → index
│   │   ├── hybrid_search.py     # BM25 + dense + RRF fusion
│   │   └── reranker.py          # Cross-encoder re-ranking
│   ├── api/                     # Phase 2: FastAPI
│   │   ├── main.py              # App + /health
│   │   └── routes/ingest.py     # POST /ingest
│   └── utils/token_cost.py      # Token/cost summary for coordinator CLI
├── scripts/
│   ├── run_coordinator.py       # Phase 1 CLI
│   ├── download_seed_docs.py    # Phase 2: download 10 ArXiv PDFs
│   ├── seed_knowledge_base.py   # Phase 2: index all seed PDFs
│   └── query_knowledge_base.py  # Phase 2: search the KB from terminal
├── data/
│   ├── sample_docs/
│   │   ├── manifest.json        # List of 10 seed papers
│   │   └── pdfs/                # Downloaded PDFs (gitignored)
│   ├── bm25_index.json          # Sparse index (gitignored, rebuilt by seed)
│   └── qdrant_storage/          # Dense vectors (gitignored, local Qdrant)
├── tests/                       # pytest unit tests
└── docs/
    └── build_project_running.md # This file
```

---

# Phase 1 — Coordinator & LangGraph scaffold

## Goal

Take a broad research question and return a **structured plan** with 2–4 sub-tasks — without answering the question yet.

Example input:

```
"What are the latest advances in agentic RAG?"
```

Example output (JSON):

```json
{
  "original_query": "What are the latest advances in agentic RAG?",
  "sub_tasks": [
    {
      "id": 1,
      "description": "Survey recent papers on agentic RAG architectures",
      "rationale": "Establish academic foundations"
    },
    {
      "id": 2,
      "description": "Compare commercial agentic RAG products",
      "rationale": "Capture industry adoption"
    }
  ],
  "research_strategy": "Start with papers, then industry, then limitations"
}
```

---

## Step-by-step: what happens when you run the coordinator

**Command:**

```bash
python scripts/run_coordinator.py "What are the latest advances in agentic RAG?"
```

### Step 1 — Script starts (`scripts/run_coordinator.py`)

1. Loads `.env` (OpenAI API key, model name).
2. Reads your question from the command line.
3. Calls `plan_research_with_usage(query)`.

### Step 2 — Coordinator agent runs (`src/agents/coordinator.py`)

Inside `plan_research_with_usage()`:

1. **Loads settings** — model `gpt-5.4-mini`, API key from `.env`.
2. **Creates the LLM** — `ChatOpenAI(...)`.
3. **Forces structured output** — this is critical:

   ```python
   structured_llm = llm.with_structured_output(ResearchPlan, include_raw=True)
   ```

   The model must return JSON matching the `ResearchPlan` schema — not free-form text.

4. **Sends two messages to OpenAI:**
   - **System message** — rules: plan only, 2–4 sub-tasks, do not answer the question.
   - **Human message** — your actual question.

5. **OpenAI returns JSON** → Pydantic validates it → you get a `ResearchPlan` object.
6. **Token usage** is extracted from the raw response and cost is estimated (`src/utils/token_cost.py`).
7. **Prints** the plan JSON + token/cost summary.

### Step 4 — Token & cost summary (added after initial Phase 1)

After the JSON plan, the CLI prints something like:

```
--- Token & cost summary (gpt-5.4-mini) ---
Input tokens:  412
Output tokens: 186
Total tokens:  598
Est. input cost:  $0.000062
Est. output cost: $0.000112
Est. total cost:  $0.000174
(Costs are estimates — verify against OpenAI/LangSmith billing.)
```

**How it works:**
- `include_raw=True` on structured output returns the raw `AIMessage` with `usage_metadata`
- `src/utils/token_cost.py` extracts input/output tokens and estimates USD cost
- Override pricing via `OPENAI_INPUT_PRICE_PER_1M` / `OPENAI_OUTPUT_PRICE_PER_1M` in `.env`

**LangSmith vs local summary:**

| | Local CLI summary | LangSmith (optional) |
|--|-------------------|----------------------|
| Setup | Built-in | `LANGCHAIN_TRACING_V2=true` + API key |
| Use case | Instant feedback in terminal | Full trace history, per-agent breakdown in Phase 3 |
| Cost tracking | Estimated from price table | Dashboard for supported models |

Both can coexist; LangSmith is recommended for capstone observability long-term.

### Step 5 — The three output keys explained

Defined in `src/models/schemas.py`:

| Key | Meaning |
|-----|---------|
| `original_query` | Your exact question |
| `sub_tasks` | 2–4 smaller jobs (each has `id`, `description`, `rationale`) |
| `research_strategy` | One paragraph explaining the overall approach |

Pydantic enforces **exactly 2–4 sub-tasks**. Wrong shape = validation error, not silent failure.

---

## Why the coordinator has no search tools

The coordinator **plans only**. It does not search the web or the knowledge base.

Later:
- **Web Researcher** (Phase 3) → Tavily, ArXiv, live web
- **Doc Analyst** (Phase 3) → searches the Phase 2 knowledge base
- **Report Writer** (Phase 4) → combines everything into a report

This separation makes the system debuggable and matches the capstone architecture.

---

## LangGraph orchestration (Phase 1 skeleton)

**Command:**

```bash
python scripts/run_coordinator.py --graph "Competitive landscape for AI coding assistants"
```

### Flow diagram

```
User Query
    │
    ▼
┌─────────────┐
│ Coordinator │  ← real LLM, produces sub_tasks
└──────┬──────┘
       │  Send() API — one worker per sub-task, in parallel
       ▼
┌─────────────────┐
│ research_worker │  ← Phase 1 STUB (returns empty findings)
└────────┬────────┘
         ▼
┌─────────────────┐
│ report_writer   │  ← Phase 1 STUB ("Phase 1 stub — not implemented")
└────────┬────────┘
         ▼
        END
```

> **Note (Phase 3):** Stub workers were replaced. The live graph now dispatches `web_researcher` and `doc_analyst` in parallel per sub-task, then runs a real `report_writer`. See [Phase 3](#phase-3--specialist-agents--full-pipeline).

### Key files

| File | Role |
|------|------|
| `src/graph/state.py` | `AgentState` — shared dict: `query`, `sub_tasks`, `web_findings`, `doc_findings`, etc. |
| `src/graph/nodes.py` | `coordinator_node` calls `plan_research()`; other nodes are stubs |
| `src/graph/graph.py` | Builds `StateGraph`, wires edges, uses `Send()` for parallel dispatch |

The `--graph` mode now runs the **full Phase 3 pipeline** (not stubs).

---

## Phase 1 files created (summary)

| File | Purpose |
|------|---------|
| `pyproject.toml` | Dependencies: langgraph, langchain-openai, pydantic, pytest |
| `src/config/settings.py` | Typed environment variables |
| `src/models/schemas.py` | `ResearchPlan`, `SubTask`, `AgentState` types |
| `src/agents/coordinator.py` | Structured-output planning agent |
| `src/graph/*` | LangGraph state, nodes, compiled graph |
| `scripts/run_coordinator.py` | CLI for coordinator-only or full graph |
| `src/utils/token_cost.py` | Local token + cost summary after each LLM call |
| `tests/test_coordinator.py`, `tests/test_graph.py` | Unit tests |

---

# Phase 2 — RAG knowledge base

## Goal

Build a **curated document library** that can be searched with advanced retrieval — separate from live web search.

Key design choice: **index documents upfront**, do not rebuild the index from the user's query at runtime.

```
INGEST TIME (before research)     QUERY TIME (during research)
─────────────────────────────     ────────────────────────────
Download PDFs                     User / agent asks a question
Load & chunk                        hybrid_search(query)
Embed & store                       Return top 5 relevant chunks
```

---

## Architecture: dual-index RAG

We store every chunk in **two indexes**:

| Index | Technology | Stored where | Good at |
|-------|------------|--------------|---------|
| **Dense** | OpenAI `text-embedding-3-small` + Qdrant | `data/qdrant_storage/` | Meaning, paraphrases |
| **Sparse** | BM25 keyword search | `data/bm25_index.json` | Exact terms ("self-attention", "ViT") |

At query time both indexes run, results are merged with **RRF (Reciprocal Rank Fusion)**, then **cross-encoder re-ranking** picks the best 5 chunks.

```
Query
  ├──► Dense search (Qdrant)  ──┐
  │                              ├──► RRF fusion ──► Cross-encoder rerank ──► Top 5 chunks
  └──► BM25 search (JSON)   ──┘
```

---

## Step-by-step: seeding the knowledge base

### Step 1 — Download PDFs

**Command:**

```bash
python scripts/download_seed_docs.py
```

**What it does:**

1. Reads `data/sample_docs/manifest.json` (10 ArXiv papers).
2. Downloads each PDF from `https://arxiv.org/pdf/{arxiv_id}.pdf`.
3. Saves to `data/sample_docs/pdfs/` (skips if already downloaded).

**The 10 papers:**

| # | Paper | Category |
|---|-------|----------|
| 1 | Attention Is All You Need | transformers |
| 2 | BERT | transformers |
| 3 | GPT-3 | llms |
| 4 | LLaMA | llms |
| 5 | InstructGPT | llms |
| 6 | Vision Transformer (ViT) | vision |
| 7 | CLIP | vision |
| 8 | Segment Anything (SAM) | vision |
| 9 | Mamba | architecture |
| 10 | GPT-4 Technical Report | llms |

### Step 2 — Index all PDFs

**Command:**

```bash
python scripts/seed_knowledge_base.py
```

**What happens for each PDF** (`src/rag/ingestion.py`):

```
PDF file
  │
  ▼
PyPDFLoader          ← extract text from each page
  │
  ▼
SemanticChunker      ← split into coherent chunks (uses OpenAI embeddings)
  │                   (falls back to fixed-size split if semantic fails)
  ▼
OpenAI embed         ← text-embedding-3-small, one vector per chunk
  │
  ├──► Qdrant upsert     ← dense index (data/qdrant_storage/)
  └──► BM25 append       ← sparse index (data/bm25_index.json)
```

**After a successful run:** ~1,380 chunks indexed (exact count depends on PDF sizes).

**Deduplication:** Re-running seed on the same PDFs adds 0 new chunks (same `chunk_id` = skip).

---

## Step-by-step: searching the knowledge base

**Command:**

```bash
python scripts/query_knowledge_base.py "self-attention mechanism in transformers"
```

### Inside `hybrid_search()` (`src/rag/hybrid_search.py`)

1. **Dense search** — embed query, find nearest vectors in Qdrant (`index_store.py`).
2. **BM25 search** — keyword score all chunks in `bm25_index.json`.
3. **RRF fusion** — merge rankings:

   ```
   score(chunk) += 1 / (60 + rank_in_dense)
   score(chunk) += 1 / (60 + rank_in_sparse)
   ```

4. **Cross-encoder rerank** — `ms-marco-MiniLM-L-6-v2` scores query+chunk pairs (`reranker.py`).
5. Return top 5 `DocumentChunk` objects (text, source, title, score).

**Fast mode (skip reranker):**

```bash
python scripts/query_knowledge_base.py "vision transformer patches" --skip-rerank
```

**Example good result:** Query about self-attention → top chunk from `1706.03762_attention.pdf`.

### Why `query_knowledge_base.py` exists (not just Python heredocs)

Early testing used inline Python:

```bash
python << 'EOF'
from dotenv import load_dotenv
load_dotenv()   # ← fails with AssertionError in heredoc context
...
EOF
```

**Problems discovered:**
1. `load_dotenv()` without a path fails in heredoc (`AssertionError: frame.f_back is not None`)
2. Creating two `IndexStore()` instances locks local Qdrant (`Storage folder already accessed`)

**Fix:** Use the dedicated script (single store, explicit `.env` path):

```bash
python scripts/query_knowledge_base.py "your query here"
python scripts/query_knowledge_base.py "your query" --skip-rerank -k 3 --json
```

### Verify index after seeding

```bash
# Chunk count from BM25 metadata
python -c "
import json
from pathlib import Path
chunks = json.loads(Path('data/bm25_index.json').read_text())['chunks']
print('BM25 chunks:', len(chunks))
"

# List local Qdrant files
ls -la data/qdrant_storage/
```

You should see ~1,380 chunks after a full seed run.

---

## Local Qdrant (no Docker required)

Default in `.env`:

```bash
QDRANT_MODE=local
QDRANT_LOCAL_PATH=data/qdrant_storage
```

Vectors are stored as files on disk. No `docker compose` needed.

For Docker/server mode (optional):

```bash
QDRANT_MODE=server
QDRANT_URL=http://localhost:6333
docker compose up -d
```

---

## FastAPI layer (Phase 2)

**Start server:**

```bash
uvicorn src.api.main:app --reload --port 8000
```

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness check → `{"status": "ok"}` |
| `POST` | `/ingest` | Add PDF, URL, or text to knowledge base |

Interactive docs: http://localhost:8000/docs

### POST /ingest — rules

Provide **exactly one** of: `text`, `url`, `pdf_path`.

**Example — ingest text:**

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Vision transformers apply self-attention to image patches.",
    "title": "ViT Notes",
    "category": "vision"
  }'
```

**Response:**

```json
{
  "chunks_indexed": 1,
  "source": "ViT Notes",
  "message": "Successfully indexed 1 chunks"
}
```

Same ingestion pipeline as the seed script — chunk, embed, dual index.

**Note:** There is no `/search` API yet. Search from terminal via `query_knowledge_base.py`. Phase 3 wires search into the Doc Analyst agent.

---

# API testing walkthrough

This section documents the API testing flow used after `uvicorn src.api.main:app --reload --port 8000` was running and `/health` returned `ok`.

### Step 1 — Health check

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

Expected: `{"status": "ok"}`

Also try: http://localhost:8000/docs (Swagger UI — try `/ingest` from the browser).

### Step 2 — Ingest plain text (fastest API test)

```bash
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Vision transformers apply self-attention to image patches instead of CNN convolutions.",
    "title": "ViT Notes",
    "category": "vision"
  }' | python -m json.tool
```

Expected: `"chunks_indexed": 1` or more.

### Step 3 — Confirm ingest updated the index

In a **second terminal** (keep the server running):

```bash
python scripts/query_knowledge_base.py "ViT Notes vision transformers" -k 2 --skip-rerank
```

You should see your ingested text in the results.

### Step 4 — Ingest a local PDF (absolute path)

```bash
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d "{
    \"pdf_path\": \"/Users/relanto/CLAUDE WORK/capstone project/researchiq/data/sample_docs/pdfs/1706.03762_attention.pdf\",
    \"title\": \"Attention Is All You Need\",
    \"category\": \"transformers\"
  }" | python -m json.tool
```

If already seeded: `"chunks_indexed": 0` (deduplication — expected).

### Step 5 — Validation errors (should return HTTP 400)

Both `text` and `url` in one request:

```bash
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "hello", "url": "https://example.com"}' | python -m json.tool
```

Empty body / no field:

```bash
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{}' | python -m json.tool
```

### Step 6 — Automated API tests (no server needed)

```bash
pytest tests/test_rag.py::test_ingest_api_text tests/test_rag.py::test_ingest_api_validation_error -v
```

Uses FastAPI `TestClient` with mocked ingestion where appropriate.

### API testing checklist

| Step | Action | Pass criteria |
|------|--------|---------------|
| 1 | `GET /health` | `{"status":"ok"}` |
| 2 | `POST /ingest` with `text` | `chunks_indexed >= 1` |
| 3 | `query_knowledge_base.py` | Result contains ingested text |
| 4 | Bad payload | HTTP 400 |
| 5 | Re-ingest same PDF | `chunks_indexed: 0` |

### What the API does **not** expose yet

| Endpoint | Status |
|----------|--------|
| `GET /search?q=...` | Not built — use `query_knowledge_base.py` |
| `POST /research` | Phase 4 |

---

## Phase 2 files created (summary)

| File | Purpose |
|------|---------|
| `src/rag/index_store.py` | Qdrant client + BM25 JSON persistence |
| `src/rag/ingestion.py` | PDF/URL/text loaders, chunking, indexing |
| `src/rag/hybrid_search.py` | BM25 + dense + RRF |
| `src/rag/reranker.py` | Cross-encoder re-ranking |
| `src/api/main.py` | FastAPI app |
| `src/api/routes/ingest.py` | POST /ingest |
| `data/sample_docs/manifest.json` | 10-paper seed corpus metadata |
| `scripts/download_seed_docs.py` | ArXiv PDF downloader |
| `scripts/seed_knowledge_base.py` | Bulk indexer |
| `scripts/query_knowledge_base.py` | Terminal search CLI |
| `tests/test_rag.py` | RAG + API unit tests |

---

# Phase 3 — Specialist agents & full pipeline

Phase 3 replaces stub workers with real agents and adds a report writer.

## Architecture

```
User query
    │
    ▼
Coordinator ──► sub_tasks: ["Task A", "Task B", ...]
    │
    ├──► Web Researcher (× N tasks) ──► Tavily search + optional page fetch
    │
    └──► Doc Analyst (× N tasks) ──► hybrid_search(task) on Phase 2 KB
                │
                ▼
         Report Writer ──► structured ResearchReport + markdown synthesis
```

For each sub-task, LangGraph dispatches **two** parallel `Send()` calls — one to `web_researcher`, one to `doc_analyst`. Findings accumulate via reducers on `web_findings` and `doc_findings`. When all workers finish, `report_writer` synthesizes a markdown report.

## Key files

| File | Purpose |
|------|---------|
| `src/tools/search_tools.py` | Tavily web search with retries |
| `src/tools/scraper_tools.py` | HTTP page fetch + HTML-to-text |
| `src/tools/rag_tools.py` | LangChain wrapper for `hybrid_search()` |
| `src/mcp/tavily_server.py` | MCP server exposing `web_search` + `get_page_content` |
| `src/agents/web_researcher.py` | ReAct agent with Tavily tools |
| `src/agents/doc_analyst.py` | Calls `hybrid_search()` per sub-task |
| `src/agents/report_writer.py` | Structured `ResearchReport` → markdown |
| `scripts/run_research.py` | End-to-end research CLI |
| `tests/test_agents.py` | Agent unit tests |

## Prerequisites

1. Phase 2 knowledge base seeded (`python scripts/seed_knowledge_base.py`)
2. `.env` contains `OPENAI_API_KEY` and `TAVILY_API_KEY`

## Run the full pipeline

```bash
python scripts/run_research.py "What are the latest advances in agentic RAG?"
```

Or via the coordinator script:

```bash
python scripts/run_coordinator.py --graph "Compare transformer and Mamba architectures"
```

**What happens:**

1. **Coordinator** decomposes the query into 2–4 sub-tasks.
2. **Web Researcher** runs a ReAct loop per sub-task (Tavily search; optional page scrape).
3. **Doc Analyst** runs hybrid RAG per sub-task against the seeded corpus.
4. **Report Writer** synthesizes all findings into a cited markdown report.

## Tavily MCP server (standalone)

The MCP server can run independently for tooling demos:

```bash
python src/mcp/tavily_server.py
```

The web researcher uses embedded LangChain tools by default (`use_mcp=False`). Pass `use_mcp=True` to `research_web()` to load tools from the MCP subprocess instead.

## Tests

```bash
pytest tests/test_agents.py tests/test_graph.py -v
```

Integration tests (call live APIs) are marked `@pytest.mark.integration` and skipped by default.

---

# How Phase 1, 2, and 3 connect

Phase 3 **connects** Phase 1 planning with Phase 2 retrieval and adds live web search:

```
User query
    │
    ▼
Coordinator (Phase 1) ──► sub_tasks: ["Task A", "Task B", "Task C"]
    │
    ├──► Web Researcher (Phase 3) ──► Tavily, live web
    │
    └──► Doc Analyst (Phase 3) ──► hybrid_search(task) ──► Phase 2 KB
                │
                ▼
         Report Writer (Phase 3) ──► markdown report
```

Phase 2 built the **library**. Phase 1 built the **planner**. Phase 3 gives the **researchers** access to both and produces the first real report.

---

# Environment & configuration

Copy and edit:

```bash
cp .env.example .env
```

**Required for Phase 1:**

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.4-mini
```

**Required for Phase 2 (embeddings + search):**

```bash
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
QDRANT_MODE=local
QDRANT_LOCAL_PATH=data/qdrant_storage
BM25_INDEX_PATH=data/bm25_index.json
```

**Optional:**

```bash
USER_AGENT=ResearchIQ/0.2          # silences web-loader warning
OPENAI_INPUT_PRICE_PER_1M=0.15       # custom cost estimates for CLI summary
OPENAI_OUTPUT_PRICE_PER_1M=0.60
TAVILY_API_KEY=...                   # Phase 3 web search (required for full pipeline)
```

---

# Commands cheat sheet

## Setup (once)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env — set OPENAI_API_KEY
```

## Phase 1 — Coordinator

```bash
# Planning only
python scripts/run_coordinator.py "What are the latest advances in agentic RAG?"

# Full graph (same as run_research.py)
python scripts/run_coordinator.py --graph "Competitive landscape for AI coding assistants"
```

## Phase 3 — Full research pipeline

```bash
# End-to-end: coordinator → web + doc agents → report writer
python scripts/run_research.py "What are the latest advances in agentic RAG?"

# Same pipeline via coordinator script
python scripts/run_coordinator.py --graph "Compare GPT-4 and LLaMA capabilities"
```

## Phase 2 — Knowledge base

```bash
# Download seed PDFs (once)
python scripts/download_seed_docs.py

# Index all 10 papers (requires OPENAI_API_KEY)
python scripts/seed_knowledge_base.py

# Search the KB
python scripts/query_knowledge_base.py "self-attention mechanism in transformers"
python scripts/query_knowledge_base.py "vision transformer patches" --skip-rerank -k 3

# Start API
uvicorn src.api.main:app --reload --port 8000

# Ingest via API
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Your note here.", "title": "My Note", "category": "notes"}'
```

## Tests

```bash
pytest tests/ -v -m "not integration"
```

## Smoke-test queries (verify retrieval quality)

| Query | Expected top document |
|-------|----------------------|
| `self-attention mechanism in transformers` | Attention Is All You Need |
| `bidirectional pre-training for NLP` | BERT |
| `vision transformer patch embeddings` | ViT |
| `image text contrastive learning` | CLIP |
| `linear time sequence modeling alternative to transformers` | Mamba |

---

# What comes next (Phase 4+)

| Phase | Scope |
|-------|-------|
| **Phase 4** | Notion MCP, `POST /research` streaming API, Streamlit UI, Langfuse |

---

# Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `command not found: docker` | Docker not installed on Mac | Use default `QDRANT_MODE=local` — no Docker needed |
| `fatal: couldn't find remote ref #` | Inline `# comment` on same line as git command | Put comments on separate lines; run git without trailing `# ...` |
| Git push rejected (non-fast-forward) | Phase 1 commit was orphan root, not on top of remote | `git fetch origin && git rebase origin/main`, resolve README conflicts, push |
| `AssertionError` from `load_dotenv()` | Heredoc `python << 'EOF'` | Use `scripts/query_knowledge_base.py` or `load_dotenv(".env")` |
| `AttributeError: no attribute 'search'` | Old Qdrant API | Fixed: uses `query_points()` in `index_store.py` |
| `Storage folder already accessed` | Multiple Qdrant local clients on same path (parallel doc analysts) | Fixed: `get_index_store()` singleton + lock; stop other processes using the same path |
| `chunks_indexed: 0` on re-ingest | Deduplication by `chunk_id` | Expected for already-indexed docs |
| `USER_AGENT not set` warning | Web/HTTP library identity (URL ingest, HF) | Add `USER_AGENT=ResearchIQ/0.2` to `.env`; harmless for PDF-only seed |
| Coordinator works without remembering API key | Key is in `.env` | `grep OPENAI_API_KEY .env` — gitignored, not on GitHub |
| OpenAI auth error on search/seed | Missing or invalid key | Set `OPENAI_API_KEY` in `.env` |
| Slow first hybrid search | Cross-encoder model download (~80MB) | Use `--skip-rerank`; later queries faster |
| `403 Forbidden` on page fetch | Site blocks scrapers (e.g. openai.com) | Expected — agent continues with Tavily snippets; no crash |
| `Exception ignored in QdrantClient.__del__` | Local Qdrant client not closed before exit | Fixed: `close_index_store()` runs in `run_research()` and via `atexit` |
| Seed takes long + costs money | Semantic chunking + embeddings for 10 full PDFs | Normal; one-time cost |

### Git workflow reference

```bash
# Clone and setup
git clone https://github.com/abanik12/MultiAgent-ResearchIQ.git
cd MultiAgent-ResearchIQ
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

# Feature branch example (token/cost feature was merged this way)
git checkout -b cursor/your-feature-name
# ... make changes ...
git add .
git commit -m "Your message"
git checkout main
git merge cursor/your-feature-name
git push origin main
```

---

# Capstone talking points

1. **Multi-agent planning** — Coordinator decomposes queries with structured Pydantic output (2–4 sub-tasks).
2. **Advanced RAG** — Hybrid BM25 + dense retrieval, RRF fusion, cross-encoder rerank (not naive vector search).
3. **Curated KB** — 10 foundational AI papers, reproducible via manifest + scripts.
4. **Production patterns** — FastAPI, typed settings, dual-index persistence, local + Docker Qdrant options.
5. **Separation of concerns** — Planner (Phase 1) vs library (Phase 2) vs live web + synthesis (Phase 3).
6. **Observability options** — Local token/cost CLI + optional LangSmith tracing.
7. **Pragmatic engineering** — Local Qdrant when Docker unavailable; dedicated scripts over fragile heredocs.

---

## Current operational status (your machine)

As of the last build session:

| Component | Status |
|-----------|--------|
| Phase 1 coordinator CLI | Working |
| Phase 1 `--graph` mode | Working (full pipeline) |
| Phase 3 `run_research.py` | Working |
| Web Researcher (Tavily) | Working |
| Doc Analyst (hybrid RAG) | Working |
| Report Writer (markdown) | Working |
| 10 seed PDFs downloaded | Done |
| Knowledge base indexed | Done (~1,380 chunks) |
| `query_knowledge_base.py` | Working |
| FastAPI `/health` | Working |
| FastAPI `/ingest` | Working |
| Docker / `docker compose` | Not required (local Qdrant) |
| GitHub `main` branch | Phase 1 + Phase 2 + token/cost merged |

---

*Last updated: Phase 3 complete — specialist agents, Tavily MCP, report writer, and full research pipeline.*
