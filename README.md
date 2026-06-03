# MultiAgent-ResearchIQ

**ResearchIQ** is an autonomous AI Research & Competitive Intelligence Agent that takes a topic or question and produces a structured intelligence report.

## What it does

- Accepts a natural-language research query
- Decomposes it into focused sub-tasks via a Coordinator agent
- Dispatches parallel Web Researcher and Document Analyst agents
- Synthesises findings into a cited, structured report saved to Notion

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (multi-agent, async) |
| LLM | Claude Sonnet 4.6 + GPT-4o fallback |
| Vector Store | Qdrant (Docker) |
| Retrieval | Hybrid BM25 + dense + RRF + cross-encoder re-ranking |
| MCP Servers | Custom Tavily MCP + Notion MCP |
| Backend | FastAPI + asyncio |
| Frontend | Streamlit |
| Observability | LangSmith + Langfuse |

## Documentation

- [`docs/ResearchIQ_System_Design.docx`](docs/ResearchIQ_System_Design.docx) — Full system & solution design document

## Project Status

🚧 In active development — Capstone Project, AI Engineering Cohort
