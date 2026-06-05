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
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "researchiq"

    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None
    langchain_project: str = "researchiq"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
