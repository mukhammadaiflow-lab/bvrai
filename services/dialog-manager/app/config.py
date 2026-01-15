"""
Configuration module for Dialog Manager Service.

All secrets are read from environment variables.
"""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Server configuration
    port: int = 3003
    host: str = "0.0.0.0"
    environment: Literal["development", "production", "test"] = "development"
    debug: bool = False

    # LLM Configuration
    # TODO: Add your LLM API credentials here
    llm_provider: Literal["mock", "openai", "anthropic"] = "mock"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_model: str = "gpt-4"  # or "claude-3-opus-20240229"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.7

    # Vector DB Configuration
    # TODO: Add Pinecone credentials for production
    vector_db_provider: Literal["local", "pinecone"] = "local"
    vector_db_path: str = "./data/vectors.db"
    pinecone_api_key: str = ""
    pinecone_environment: str = ""
    pinecone_index_name: str = "dialog-manager"

    # RAG Configuration
    rag_top_k: int = 5
    rag_similarity_threshold: float = 0.7

    # Session Configuration
    session_history_max_turns: int = 50  # Increased for better conversation context
    session_ttl_seconds: int = 86400  # 24 hours (was 1 hour - too aggressive)

    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60

    # CORS
    cors_origins: str = "*"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
