"""FastAPI application for ResearchIQ."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import ingest


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="ResearchIQ API",
    description="Autonomous multi-agent research platform",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(ingest.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
