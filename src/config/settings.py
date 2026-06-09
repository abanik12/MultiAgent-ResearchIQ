from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_model: str = "gpt-5.4-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_input_price_per_1m: float | None = None
    openai_output_price_per_1m: float | None = None

    tavily_api_key: str | None = None
    web_search_recent_limit_enabled: bool = False
    web_search_recent_limit: int = 30  # max total web links across all sub-tasks when enabled
    qdrant_mode: str = "local"  # "local" = embedded file storage (no Docker); "server" = Qdrant via URL
    qdrant_url: str = "http://localhost:6333"
    qdrant_local_path: str = "data/qdrant_storage"
    qdrant_collection: str = "researchiq"
    bm25_index_path: str = "data/bm25_index.json"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rag_skip_rerank: bool = False
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 100

    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None
    langchain_project: str = "researchiq"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
