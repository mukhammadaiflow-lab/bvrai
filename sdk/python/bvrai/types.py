"""
Type definitions for the BVRAI SDK.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar


T = TypeVar("T")


@dataclass
class PaginatedResponse(Generic[T]):
    """Paginated API response."""
    items: List[T]
    total: int
    page: int
    page_size: int

    @property
    def total_pages(self) -> int:
        return (self.total + self.page_size - 1) // self.page_size

    @property
    def has_next(self) -> bool:
        return self.page < self.total_pages

    @property
    def has_prev(self) -> bool:
        return self.page > 1


@dataclass
class Agent:
    """Voice agent configuration."""
    id: str
    name: str
    organization_id: str = ""
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    voice_config_id: Optional[str] = None
    llm_config: Optional[Dict[str, Any]] = None
    tools: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "draft"
    version: int = 1
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def __post_init__(self):
        if self.llm_config is None:
            self.llm_config = {}


@dataclass
class Call:
    """Voice call record."""
    id: str
    organization_id: str = ""
    agent_id: str = ""
    conversation_id: Optional[str] = None
    external_call_id: Optional[str] = None
    direction: str = "inbound"
    from_number: Optional[str] = None
    to_number: Optional[str] = None
    status: str = "initiated"
    hangup_reason: Optional[str] = None
    recording_url: Optional[str] = None
    started_at: Optional[str] = None
    answered_at: Optional[str] = None
    ended_at: Optional[str] = None
    duration_seconds: Optional[int] = None
    cost_cents: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class Message:
    """Conversation message."""
    id: str
    conversation_id: str = ""
    role: str = "user"
    content: str = ""
    audio_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    latency_ms: Optional[float] = None
    token_count: Optional[int] = None
    created_at: Optional[str] = None


@dataclass
class Conversation:
    """Voice conversation."""
    id: str
    organization_id: str = ""
    agent_id: str = ""
    call_id: Optional[str] = None
    customer_phone: Optional[str] = None
    customer_id: Optional[str] = None
    channel: str = "phone"
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)
    messages: List[Message] = field(default_factory=list)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    duration_seconds: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def __post_init__(self):
        if self.messages and isinstance(self.messages[0], dict):
            self.messages = [Message(**m) for m in self.messages]


@dataclass
class VoiceConfiguration:
    """Voice configuration for STT/TTS."""
    id: str
    organization_id: str = ""
    name: str = ""
    description: Optional[str] = None
    stt_provider: str = "deepgram"
    stt_config: Dict[str, Any] = field(default_factory=dict)
    tts_provider: str = "elevenlabs"
    tts_config: Dict[str, Any] = field(default_factory=dict)
    voice_id: Optional[str] = None
    language: str = "en"
    speed: float = 1.0
    pitch: float = 1.0
    is_default: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class Voice:
    """TTS voice option."""
    id: str
    name: str
    provider: str
    language: str = "en"
    gender: str = "neutral"
    preview_url: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class Webhook:
    """Webhook configuration."""
    id: str
    organization_id: str = ""
    name: str = ""
    url: str = ""
    events: List[str] = field(default_factory=list)
    secret: str = ""
    is_active: bool = True
    headers: Dict[str, str] = field(default_factory=dict)
    last_triggered_at: Optional[str] = None
    failure_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class WebhookDelivery:
    """Webhook delivery attempt."""
    id: str
    webhook_id: str
    event_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    response_status: Optional[int] = None
    response_body: Optional[str] = None
    duration_ms: Optional[float] = None
    success: bool = False
    created_at: Optional[str] = None


@dataclass
class AnalyticsSummary:
    """Analytics summary data."""
    period: str = ""
    total_calls: int = 0
    total_duration_minutes: float = 0.0
    average_duration_seconds: float = 0.0
    successful_calls: int = 0
    failed_calls: int = 0
    success_rate: float = 0.0
    total_cost_cents: int = 0
    unique_callers: int = 0


@dataclass
class UsageSummary:
    """Usage summary for billing."""
    period: str = ""
    calls: int = 0
    minutes: float = 0.0
    stt_characters: int = 0
    tts_characters: int = 0
    llm_tokens: int = 0
    estimated_cost_cents: int = 0


@dataclass
class Organization:
    """Organization account."""
    id: str
    name: str
    slug: str = ""
    plan: str = "free"
    is_active: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class User:
    """User account."""
    id: str
    organization_id: str = ""
    email: str = ""
    name: Optional[str] = None
    role: str = "member"
    is_active: bool = True
    last_login: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
