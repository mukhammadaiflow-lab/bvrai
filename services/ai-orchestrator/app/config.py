"""Configuration for AI Orchestrator."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Service settings
    service_name: str = "ai-orchestrator"
    host: str = "0.0.0.0"
    port: int = 8085
    debug: bool = False
    log_level: str = "info"

    # LLM Provider settings
    llm_provider: Literal["openai", "anthropic", "groq", "mock"] = "openai"

    # OpenAI settings
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = "gpt-4o-mini"  # Fast and capable
    openai_max_tokens: int = 500  # Keep responses concise for voice
    openai_temperature: float = 0.7

    # Anthropic settings
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    anthropic_model: str = "claude-3-haiku-20240307"  # Fast for voice
    anthropic_max_tokens: int = 500

    # Groq settings (fastest)
    groq_api_key: str = Field(default="", description="Groq API key")
    groq_model: str = "llama-3.1-8b-instant"  # Very fast

    # RAG settings
    rag_enabled: bool = True
    rag_top_k: int = 3
    rag_score_threshold: float = 0.7

    # Embedding settings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Qdrant settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "knowledge_base"

    # Prompt settings
    default_system_prompt: str = """You are a helpful AI voice assistant. Keep your responses concise and conversational, as they will be spoken aloud.

Guidelines:
- Be friendly and professional
- Keep responses under 2-3 sentences unless more detail is needed
- Avoid using markdown, lists, or formatting (it's spoken aloud)
- If you don't know something, say so
- Ask clarifying questions when needed"""

    # Rate limiting
    max_concurrent_requests: int = 50
    request_timeout: int = 30

    # Caching
    cache_enabled: bool = True
    cache_ttl: int = 3600


@lru_cache
def get_settings() -> Settings:
    """Get cached settings."""
    return Settings()
