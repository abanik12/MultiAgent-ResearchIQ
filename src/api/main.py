"""FastAPI application for ResearchIQ."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import ingest, research
from src.config.settings import get_settings
from src.utils.tracing import configure_langsmith


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_langsmith(get_settings())
    yield


app = FastAPI(
    title="ResearchIQ API",
    description="Autonomous multi-agent research platform",
    version="0.5.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(research.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
