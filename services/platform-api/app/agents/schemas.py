"""Agent Pydantic schemas."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.database.models import AgentStatus


class ToolSchema(BaseModel):
    """Tool/function schema."""
    name: str
    description: str
    parameters: dict = {}


class AgentBase(BaseModel):
    """Base agent schema."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None

    # Voice
    voice_id: Optional[str] = None
    voice_name: Optional[str] = None
    language: str = "en-US"

    # Behavior
    system_prompt: Optional[str] = None
    greeting_message: Optional[str] = None
    fallback_message: Optional[str] = None
    goodbye_message: Optional[str] = None

    # LLM
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=500, ge=1, le=4000)

    # Phone
    phone_number: Optional[str] = None
    forward_number: Optional[str] = None

    # Configuration
    tools: list[ToolSchema] = []
    settings: dict = {}


class AgentCreate(AgentBase):
    """Schema for creating an agent."""
    pass


class AgentUpdate(BaseModel):
    """Schema for updating an agent."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    status: Optional[AgentStatus] = None

    voice_id: Optional[str] = None
    voice_name: Optional[str] = None
    language: Optional[str] = None

    system_prompt: Optional[str] = None
    greeting_message: Optional[str] = None
    fallback_message: Optional[str] = None
    goodbye_message: Optional[str] = None

    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1, le=4000)

    phone_number: Optional[str] = None
    forward_number: Optional[str] = None

    tools: Optional[list[ToolSchema]] = None
    settings: Optional[dict] = None


class AgentResponse(AgentBase):
    """Schema for agent response."""
    id: UUID
    owner_id: UUID
    status: AgentStatus
    created_at: datetime
    updated_at: Optional[datetime] = None

    # Stats
    total_calls: int = 0
    avg_duration: float = 0

    class Config:
        from_attributes = True


class AgentListResponse(BaseModel):
    """Schema for list of agents."""
    agents: list[AgentResponse]
    total: int
    page: int
    page_size: int


class AgentStats(BaseModel):
    """Agent statistics."""
    agent_id: UUID
    total_calls: int
    completed_calls: int
    failed_calls: int
    avg_duration_seconds: float
    total_duration_seconds: int
    calls_today: int
    calls_this_week: int
    calls_this_month: int
