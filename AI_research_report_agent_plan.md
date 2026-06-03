# ResearchIQ — AI Research & Competitive Intelligence Agent
## Capstone Project Plan

---

## Overview

**Project Name:** ResearchIQ

**What it does:** An autonomous research platform that takes a topic or question and produces a structured intelligence report. It orchestrates specialist AI agents across live web search and a curated knowledge base, combining everything into a polished, saved report.

**Example use cases:**
- "What are the latest advances in agentic RAG?"
- "Competitive landscape for AI coding assistants in 2025"
- "Summarize recent papers on LLM evaluation frameworks"

**Why it impresses in interviews:** Every SaaS product, engineering, and strategy team spends hours on this kind of research manually. This system automates it end-to-end using every major AI engineering pattern: multi-agent orchestration, advanced RAG, tool calling, and MCP.

---

## Required Technologies Covered

| Requirement | How It's Met |
|---|---|
| Multi-agent with LangGraph | Coordinator + Web Researcher + Doc Analyst + Report Writer agents |
| Advanced RAG — Hybrid Search | BM25 (sparse) + Qdrant dense vector search, fused with RRF |
| Advanced RAG — Re-ranking | Cross-encoder model (`ms-marco-MiniLM-L-6-v2`) re-ranks top candidates |
| Agents & Tool Calling | Each agent has bound tools: Tavily, ArXiv, Wikipedia, scraper, Notion |
| MCP | Custom Tavily MCP server + Notion MCP server; agents connect as MCP clients |
| Production-grade | Async, Pydantic schemas, Docker, LangSmith + Langfuse observability, pytest |

---

## System Architecture

```
User Query / Topic
        │
        ▼
┌────────────────────────────────────────────────────┐
│           Coordinator Agent  (LangGraph)           │
│                                                    │
│  - Parses query into 2-4 focused sub-tasks         │
│  - Uses structured output (Pydantic) to plan       │
│  - Routes sub-tasks to specialist agents in        │
│    parallel using LangGraph Send() API             │
└────────┬───────────────────────┬───────────────────┘
         │                       │
    ┌────▼────┐           ┌──────▼──────┐
    │   Web   │           │  Document   │
    │Research │           │  Analyst    │
    │  Agent  │           │   Agent     │
    └────┬────┘           └──────┬──────┘
         │                       │
   Tools:                  Advanced RAG:
   - Tavily MCP search     - Ingest: PDFs, URLs, text
   - ArXiv search          - Semantic chunking
   - Wikipedia lookup      - Hybrid search (BM25 + Qdrant)
   - Web page scraper      - RRF fusion
                           - Cross-encoder re-ranking
         │                       │
         └──────────┬────────────┘
                    ▼
          ┌─────────────────┐
          │  Report Writer  │
          │     Agent       │
          │                 │
          │ - Synthesizes   │
          │   all findings  │
          │ - Writes report │
          │ - Saves via     │
          │   Notion MCP    │
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │  Final Report   │
          │  (Structured    │
          │   Markdown)     │
          │  Saved to:      │
          │  - Notion page  │
          │  - FastAPI resp │
          │  - Streamlit UI │
          └─────────────────┘
```

---

## Technology Stack

| Layer | Technology | Why |
|---|---|---|
| Agent Orchestration | LangGraph (StateGraph, async) | Industry standard for multi-agent |
| Primary LLM | Claude Sonnet 4.6 | Latest, production-ready |
| Fallback LLM | GPT-4o | Redundancy |
| Vector Store | Qdrant (Docker) | Production-grade, fast |
| Sparse Search | rank_bm25 | Keyword-based retrieval |
| Hybrid Fusion | Reciprocal Rank Fusion (RRF) | Best of dense + sparse |
| Re-ranking | sentence-transformers cross-encoder | State-of-art re-ranking |
| Embeddings | OpenAI text-embedding-3-small | Fast, cheap, good quality |
| Web Search | Tavily MCP server | Real-time web results |
| Output Storage | Notion MCP server | Saves polished reports |
| Backend API | FastAPI + async | Production-grade REST API |
| Frontend | Streamlit | Rapid UI, streaming-friendly |
| Observability | LangSmith + Langfuse | Tracing + monitoring |
| Config | Pydantic Settings v2 | Type-safe env management |
| Validation | Pydantic models | All inputs/outputs validated |
| Testing | pytest + pytest-asyncio | Async test support |
| Containerization | Docker + Docker Compose | Reproducible deployment |
| Dependency Mgmt | uv (fast) or poetry | Modern Python packaging |

---

## Project Structure

```
researchiq/
├── src/
│   ├── config/
│   │   └── settings.py              # Pydantic Settings — all env vars typed
│   │
│   ├── models/
│   │   └── schemas.py               # Pydantic schemas: ResearchReport,
│   │                                #   SearchResult, DocumentChunk,
│   │                                #   AgentState, API request/response
│   │
│   ├── rag/
│   │   ├── ingestion.py             # Doc loading, semantic chunking,
│   │   │                            #   dual indexing (Qdrant + BM25)
│   │   ├── hybrid_search.py         # BM25 + dense fusion via RRF
│   │   └── reranker.py              # Cross-encoder re-ranking (top-5)
│   │
│   ├── tools/
│   │   ├── search_tools.py          # Tavily, ArXiv, Wikipedia LangChain tools
│   │   └── scraper_tools.py         # URL content fetcher/cleaner
│   │
│   ├── mcp/
│   │   ├── tavily_server.py         # Custom MCP server wrapping Tavily API
│   │   │                            #   Exposes: web_search(), get_page_content()
│   │   └── notion_server.py         # MCP server for saving to Notion
│   │                                #   Exposes: create_page(), append_block()
│   │
│   ├── agents/
│   │   ├── coordinator.py           # Decomposes query → sub-tasks (structured output)
│   │   ├── web_researcher.py        # ReAct agent: Tavily MCP + scraper tools
│   │   ├── doc_analyst.py           # ReAct agent: hybrid RAG search tools
│   │   └── report_writer.py         # Synthesizes → structured report → Notion MCP
│   │
│   ├── graph/
│   │   ├── state.py                 # AgentState TypedDict for LangGraph
│   │   ├── nodes.py                 # LangGraph node functions (async)
│   │   └── graph.py                 # StateGraph definition + compilation
│   │                                #   coordinator → Send([web, doc]) → writer
│   │
│   └── api/
│       ├── main.py                  # FastAPI app, middleware, lifespan
│       └── routes/
│           ├── research.py          # POST /research — trigger research run
│           └── ingest.py            # POST /ingest — add docs to knowledge base
│
├── app/
│   └── streamlit_app.py             # Chat-style UI with streaming output
│
├── tests/
│   ├── test_rag.py                  # Test hybrid search + re-ranking pipeline
│   ├── test_agents.py               # Test individual agent tool calls
│   └── test_api.py                  # Integration tests for FastAPI endpoints
│
├── data/
│   └── sample_docs/                 # Sample PDFs/texts to seed knowledge base
│
├── docker-compose.yml               # App container + Qdrant container
├── Dockerfile                       # App image
├── pyproject.toml                   # Dependencies
├── .env.example                     # Template env file (no secrets)
└── README.md                        # Setup, architecture diagram, demo GIF
```

---

## Key Implementation Details

### 1. LangGraph State

```python
# src/graph/state.py
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from src.models.schemas import SearchResult, DocumentChunk, ResearchReport

class AgentState(TypedDict):
    query: str
    sub_tasks: list[str]
    web_findings: list[SearchResult]
    doc_findings: list[DocumentChunk]
    synthesis: str
    report: ResearchReport
    messages: Annotated[list, add_messages]
```

### 2. Parallel Agent Dispatch (Send API)

```python
# src/graph/graph.py
from langgraph.constants import Send

def route_to_agents(state: AgentState) -> list[Send]:
    return [
        Send("web_researcher", {"query": task, "task_id": i})
        for i, task in enumerate(state["sub_tasks"])
    ] + [
        Send("doc_analyst", {"query": task, "task_id": i})
        for i, task in enumerate(state["sub_tasks"])
    ]
```

### 3. Hybrid Search + Re-ranking

```python
# src/rag/hybrid_search.py
def hybrid_search(query: str, top_k: int = 5) -> list[DocumentChunk]:
    # 1. Dense search — Qdrant
    dense_results = qdrant_client.search(query_vector=embed(query), limit=20)

    # 2. Sparse search — BM25
    bm25_results = bm25_index.get_top_n(query.split(), corpus, n=20)

    # 3. RRF fusion
    fused = reciprocal_rank_fusion([dense_results, bm25_results])

    # 4. Cross-encoder re-ranking
    reranked = cross_encoder.rank(query, [r.text for r in fused[:10]])

    return reranked[:top_k]
```

### 4. Custom MCP Server (Tavily)

```python
# src/mcp/tavily_server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("tavily-research")

@server.tool()
async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for current information on a topic."""
    return await tavily_client.search(query, max_results=max_results)

@server.tool()
async def get_page_content(url: str) -> str:
    """Fetch and clean content from a web page."""
    return await scrape_and_clean(url)

async def main():
    async with stdio_server() as streams:
        await server.run(*streams, server.create_initialization_options())
```

### 5. Agents Connect to MCP as Clients

```python
# src/agents/web_researcher.py
from langchain_mcp_adapters.client import MultiServerMCPClient

async def create_web_researcher():
    client = MultiServerMCPClient({
        "tavily": {"command": "python", "args": ["src/mcp/tavily_server.py"]}
    })
    tools = await client.get_tools()
    return create_react_agent(llm, tools)
```

### 6. Production Patterns Applied

- **Async throughout**: All agents use `ainvoke`, streaming via `astream_events`
- **Pydantic everywhere**: Request/response validation, structured LLM outputs
- **Observability**: LangSmith tracing + Langfuse callbacks on all LLM calls
- **Retry logic**: `tenacity` decorators on all external API calls
- **Semantic caching**: `InMemoryCache` for repeated/similar queries
- **Rate limiting**: FastAPI middleware for API endpoint protection
- **Environment management**: All secrets in `.env`, typed via Pydantic Settings

---

## 3-Week Build Plan

### Week 1 — Foundation & RAG

**Days 1-2: Project Setup**
- Initialize repo with `uv` package manager
- Set up Pydantic Settings, `.env.example`
- Docker Compose with Qdrant
- LangSmith + Langfuse configuration

**Days 3-4: RAG Ingestion Pipeline**
- Document loaders (PDF, URL, plain text)
- Semantic chunking (`SemanticChunker` from langchain_experimental)
- Dual indexing: Qdrant (dense) + BM25 index (persist to JSON)
- `/ingest` FastAPI endpoint

**Day 5: Advanced Retrieval**
- Hybrid search with RRF fusion
- Cross-encoder re-ranking
- Unit tests for RAG pipeline (`test_rag.py`)

---

### Week 2 — Agents, Graph & MCP

**Days 6-7: LangGraph Graph**
- Define `AgentState`
- Coordinator agent with structured output (query decomposition)
- `StateGraph` skeleton with routing logic

**Days 8-9: Specialist Agents**
- Web Researcher agent (Tavily + ArXiv + Wikipedia + scraper tools)
- Document Analyst agent (hybrid RAG tools)
- Report Writer agent (synthesis + Notion MCP save)

**Day 10: MCP Integration**
- Build custom Tavily MCP server
- Build/configure Notion MCP server
- Wire agents as MCP clients via `langchain-mcp-adapters`

---

### Week 3 — Polish, Testing & Demo

**Days 11-12: FastAPI & Streaming**
- `POST /research` endpoint with async streaming response
- LangSmith trace ID returned in response headers
- Error handling, retries, rate limiting middleware

**Days 13-14: Streamlit UI**
- Chat interface with streaming output display
- Knowledge base management panel (view/add docs)
- Report history view

**Day 15: Tests, Docker & README**
- `pytest tests/` — target 80% coverage on critical paths
- Full Docker Compose build and smoke test
- README with architecture diagram, setup instructions, demo GIF
- Final end-to-end demo run: query → parallel agents → report → Notion

---

## Portfolio / Interview Talking Points

1. **Multi-agent parallelism**: Used LangGraph's `Send()` API to dispatch web and document research agents truly in parallel — not sequentially — cutting latency by ~50%.

2. **Advanced RAG beyond basics**: Implemented dual-index retrieval (BM25 sparse + Qdrant dense), fused with Reciprocal Rank Fusion, then re-ranked with a cross-encoder. Most RAG demos only do naive vector search.

3. **MCP from scratch**: Built a custom MCP server that exposes Tavily search as a standardized tool interface. Agents connect to it as MCP clients, demonstrating the full MCP protocol — not just consuming an existing server.

4. **Production engineering**: Async FastAPI backend, Pydantic validation throughout, LangSmith tracing + Langfuse monitoring, Docker deployment, retry logic with tenacity, semantic caching. This is what real AI engineering looks like vs. notebook demos.

5. **Tool calling architecture**: Each agent has a specific tool set. The coordinator does not have search tools — it only plans. This separation of concerns prevents agents from overstepping and makes the system debuggable.

---

## Environment Variables Required

```bash
# LLM
ANTHROPIC_API_KEY=
OPENAI_API_KEY=

# Search
TAVILY_API_KEY=

# Vector Store
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=researchiq

# Notion (for MCP)
NOTION_TOKEN=
NOTION_DATABASE_ID=

# Observability
LANGCHAIN_API_KEY=         # LangSmith
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=researchiq
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## Verification Steps (End-to-End Test)

1. `docker-compose up` — Qdrant starts on port 6333, app on port 8000
2. `curl -X POST localhost:8000/ingest -d '{"url": "https://arxiv.org/abs/2312.10997"}'` — ingest a paper
3. `curl -X POST localhost:8000/research -d '{"query": "Advances in RAG for enterprise use"}'` — run research
4. Watch LangSmith dashboard — should show parallel agent traces
5. Open Langfuse dashboard — LLM call latency, token counts, costs tracked
6. Check Notion workspace — report page created with sources and findings
7. `pytest tests/ -v` — all tests pass
8. Open Streamlit (`streamlit run app/streamlit_app.py`) — full UI demo

---

## Project Location

`/Users/relanto/CLAUDE WORK/capstone project/researchiq/`
