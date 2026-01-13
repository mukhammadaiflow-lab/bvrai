"""
Builder Engine Python SDK - Data Models

This module contains all the data models used throughout the SDK.
Models are implemented as dataclasses with full type hints and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from uuid import UUID
import json


class BaseModel:
    """Base class for all models with common functionality."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert model to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseModel":
        """Create model instance from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Enums
# =============================================================================

class CallStatus(str, Enum):
    """Status of a voice call."""
    QUEUED = "queued"
    RINGING = "ringing"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BUSY = "busy"
    NO_ANSWER = "no_answer"
    CANCELED = "canceled"


class CallDirection(str, Enum):
    """Direction of a call."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class AgentStatus(str, Enum):
    """Status of an agent."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DRAFT = "draft"
    ARCHIVED = "archived"


class VoiceProvider(str, Enum):
    """Voice synthesis provider."""
    ELEVENLABS = "elevenlabs"
    OPENAI = "openai"
    AZURE = "azure"
    GOOGLE = "google"
    AWS_POLLY = "aws_polly"
    DEEPGRAM = "deepgram"
    PLAYHT = "playht"
    CARTESIA = "cartesia"


class STTProvider(str, Enum):
    """Speech-to-text provider."""
    DEEPGRAM = "deepgram"
    OPENAI_WHISPER = "openai_whisper"
    AZURE = "azure"
    GOOGLE = "google"
    AWS_TRANSCRIBE = "aws_transcribe"
    ASSEMBLY_AI = "assembly_ai"


class LLMProvider(str, Enum):
    """LLM provider."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    GOOGLE = "google"
    COHERE = "cohere"
    MISTRAL = "mistral"
    GROQ = "groq"
    TOGETHER = "together"


class WebhookEvent(str, Enum):
    """Webhook event types."""
    CALL_STARTED = "call.started"
    CALL_RINGING = "call.ringing"
    CALL_ANSWERED = "call.answered"
    CALL_ENDED = "call.ended"
    CALL_FAILED = "call.failed"
    TRANSCRIPTION_READY = "transcription.ready"
    TRANSCRIPTION_PARTIAL = "transcription.partial"
    AGENT_RESPONSE = "agent.response"
    FUNCTION_CALLED = "function.called"
    RECORDING_READY = "recording.ready"
    DTMF_RECEIVED = "dtmf.received"
    CALL_TRANSFERRED = "call.transferred"
    VOICEMAIL_DETECTED = "voicemail.detected"
    HUMAN_DETECTED = "human.detected"


class CampaignStatus(str, Enum):
    """Status of a call campaign."""
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELED = "canceled"


class WorkflowTrigger(str, Enum):
    """Workflow trigger types."""
    CALL_STARTED = "call_started"
    CALL_ENDED = "call_ended"
    KEYWORD_DETECTED = "keyword_detected"
    INTENT_DETECTED = "intent_detected"
    SENTIMENT_NEGATIVE = "sentiment_negative"
    SILENCE_DETECTED = "silence_detected"
    TRANSFER_REQUESTED = "transfer_requested"
    SCHEDULE = "schedule"
    WEBHOOK = "webhook"
    API = "api"


class PhoneNumberType(str, Enum):
    """Type of phone number."""
    LOCAL = "local"
    TOLL_FREE = "toll_free"
    MOBILE = "mobile"


class PhoneNumberCapability(str, Enum):
    """Phone number capabilities."""
    VOICE = "voice"
    SMS = "sms"
    MMS = "mms"
    FAX = "fax"


# =============================================================================
# Configuration Models
# =============================================================================

@dataclass
class VoiceConfig(BaseModel):
    """Voice configuration for TTS."""
    provider: VoiceProvider = VoiceProvider.ELEVENLABS
    voice_id: str = ""
    model: Optional[str] = None
    language: str = "en-US"
    speaking_rate: float = 1.0
    pitch: float = 1.0
    volume_gain_db: float = 0.0
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = True
    optimize_streaming_latency: int = 3


@dataclass
class STTConfig(BaseModel):
    """Speech-to-text configuration."""
    provider: STTProvider = STTProvider.DEEPGRAM
    model: str = "nova-2"
    language: str = "en-US"
    punctuate: bool = True
    profanity_filter: bool = False
    diarize: bool = False
    smart_format: bool = True
    filler_words: bool = False
    interim_results: bool = True
    endpointing: int = 300
    vad_events: bool = True
    keywords: List[str] = field(default_factory=list)


@dataclass
class LLMConfig(BaseModel):
    """LLM configuration for conversation."""
    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4-turbo"
    temperature: float = 0.7
    max_tokens: int = 500
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)
    system_prompt: str = ""
    context_window: int = 8000
    response_format: Optional[str] = None


@dataclass
class FunctionDefinition(BaseModel):
    """Function/tool definition for agent."""
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str] = field(default_factory=list)
    webhook_url: Optional[str] = None
    async_execution: bool = False
    timeout_seconds: int = 30


@dataclass
class AgentConfig(BaseModel):
    """Complete agent configuration."""
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Conversation settings
    first_message: Optional[str] = None
    end_call_message: Optional[str] = None
    end_call_phrases: List[str] = field(default_factory=list)
    interruption_threshold: int = 100
    max_duration_seconds: int = 1800
    silence_timeout_seconds: int = 30
    response_delay_ms: int = 0

    # Recording
    recording_enabled: bool = True
    transcription_enabled: bool = True

    # Advanced
    voicemail_detection: bool = True
    voicemail_message: Optional[str] = None
    answering_machine_detection: bool = True
    background_sound: Optional[str] = None
    background_volume: float = 0.1

    # Functions/Tools
    functions: List[FunctionDefinition] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Core Models
# =============================================================================

@dataclass
class Agent(BaseModel):
    """AI voice agent."""
    id: str
    name: str
    description: Optional[str] = None
    status: AgentStatus = AgentStatus.ACTIVE
    config: Optional[AgentConfig] = None
    voice_id: Optional[str] = None
    phone_number_id: Optional[str] = None
    knowledge_base_ids: List[str] = field(default_factory=list)
    workflow_ids: List[str] = field(default_factory=list)
    organization_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Statistics
    total_calls: int = 0
    total_duration_seconds: int = 0
    average_duration_seconds: float = 0.0
    success_rate: float = 0.0


@dataclass
class Call(BaseModel):
    """Voice call."""
    id: str
    agent_id: str
    status: CallStatus = CallStatus.QUEUED
    direction: CallDirection = CallDirection.OUTBOUND
    from_number: Optional[str] = None
    to_number: Optional[str] = None
    duration_seconds: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    answer_time: Optional[datetime] = None
    recording_url: Optional[str] = None
    transcript_url: Optional[str] = None
    cost: float = 0.0
    cost_breakdown: Dict[str, float] = field(default_factory=dict)
    ended_reason: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    organization_id: Optional[str] = None
    created_at: Optional[datetime] = None

    # Analysis
    sentiment_score: Optional[float] = None
    summary: Optional[str] = None
    detected_intents: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)


@dataclass
class Message(BaseModel):
    """Conversation message."""
    id: str
    conversation_id: str
    role: str  # "user", "assistant", "system", "function"
    content: str
    timestamp: datetime
    audio_url: Optional[str] = None
    duration_ms: int = 0
    tokens_used: int = 0
    confidence: Optional[float] = None
    function_call: Optional[Dict[str, Any]] = None
    function_result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation(BaseModel):
    """Conversation within a call."""
    id: str
    call_id: str
    agent_id: str
    messages: List[Message] = field(default_factory=list)
    total_tokens: int = 0
    total_cost: float = 0.0
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    summary: Optional[str] = None
    sentiment: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhoneNumber(BaseModel):
    """Phone number resource."""
    id: str
    number: str
    friendly_name: Optional[str] = None
    type: PhoneNumberType = PhoneNumberType.LOCAL
    capabilities: List[PhoneNumberCapability] = field(default_factory=list)
    country_code: str = "US"
    region: Optional[str] = None
    provider: str = "twilio"
    agent_id: Optional[str] = None
    voice_url: Optional[str] = None
    sms_url: Optional[str] = None
    status_callback_url: Optional[str] = None
    monthly_cost: float = 0.0
    status: str = "active"
    organization_id: Optional[str] = None
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Voice(BaseModel):
    """Voice configuration/preset."""
    id: str
    name: str
    description: Optional[str] = None
    provider: VoiceProvider = VoiceProvider.ELEVENLABS
    provider_voice_id: str = ""
    language: str = "en-US"
    gender: Optional[str] = None
    age: Optional[str] = None
    accent: Optional[str] = None
    preview_url: Optional[str] = None
    is_custom: bool = False
    is_cloned: bool = False
    config: Optional[VoiceConfig] = None
    organization_id: Optional[str] = None
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Webhook(BaseModel):
    """Webhook configuration."""
    id: str
    url: str
    events: List[WebhookEvent] = field(default_factory=list)
    secret: Optional[str] = None
    enabled: bool = True
    description: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    retry_count: int = 3
    timeout_seconds: int = 30
    organization_id: Optional[str] = None
    created_at: Optional[datetime] = None

    # Statistics
    total_deliveries: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0
    last_triggered_at: Optional[datetime] = None
    last_status_code: Optional[int] = None


@dataclass
class Document(BaseModel):
    """Knowledge base document."""
    id: str
    knowledge_base_id: str
    name: str
    content: Optional[str] = None
    content_type: str = "text/plain"
    file_url: Optional[str] = None
    file_size: int = 0
    chunk_count: int = 0
    vector_status: str = "pending"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeBase(BaseModel):
    """Knowledge base for RAG."""
    id: str
    name: str
    description: Optional[str] = None
    document_count: int = 0
    total_chunks: int = 0
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    organization_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowAction(BaseModel):
    """Workflow action definition."""
    id: str
    type: str  # "send_sms", "send_email", "webhook", "transfer", "end_call", etc.
    config: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[str] = None
    order: int = 0


@dataclass
class Workflow(BaseModel):
    """Automated workflow."""
    id: str
    name: str
    description: Optional[str] = None
    trigger: WorkflowTrigger = WorkflowTrigger.CALL_ENDED
    trigger_config: Dict[str, Any] = field(default_factory=dict)
    actions: List[WorkflowAction] = field(default_factory=list)
    enabled: bool = True
    agent_ids: List[str] = field(default_factory=list)
    organization_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Statistics
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    last_executed_at: Optional[datetime] = None


@dataclass
class CampaignContact(BaseModel):
    """Contact in a campaign."""
    id: str
    campaign_id: str
    phone_number: str
    name: Optional[str] = None
    email: Optional[str] = None
    custom_data: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, called, completed, failed, skipped
    call_id: Optional[str] = None
    attempts: int = 0
    last_attempt_at: Optional[datetime] = None
    scheduled_at: Optional[datetime] = None


@dataclass
class Campaign(BaseModel):
    """Call campaign for batch outreach."""
    id: str
    name: str
    description: Optional[str] = None
    agent_id: str
    status: CampaignStatus = CampaignStatus.DRAFT
    contacts: List[CampaignContact] = field(default_factory=list)
    total_contacts: int = 0
    completed_contacts: int = 0
    failed_contacts: int = 0
    skipped_contacts: int = 0
    max_concurrent_calls: int = 10
    calls_per_minute: int = 5
    max_attempts: int = 3
    retry_delay_minutes: int = 60
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None
    timezone: str = "UTC"
    calling_hours_start: str = "09:00"
    calling_hours_end: str = "17:00"
    calling_days: List[str] = field(default_factory=lambda: ["mon", "tue", "wed", "thu", "fri"])
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    organization_id: Optional[str] = None
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Analytics Models
# =============================================================================

@dataclass
class CallMetrics(BaseModel):
    """Call metrics summary."""
    total_calls: int = 0
    completed_calls: int = 0
    failed_calls: int = 0
    total_duration_seconds: int = 0
    average_duration_seconds: float = 0.0
    total_cost: float = 0.0
    average_cost: float = 0.0
    success_rate: float = 0.0
    answer_rate: float = 0.0


@dataclass
class UsageMetrics(BaseModel):
    """Usage metrics."""
    period_start: datetime
    period_end: datetime
    total_calls: int = 0
    total_minutes: float = 0.0
    stt_minutes: float = 0.0
    tts_characters: int = 0
    llm_tokens: int = 0
    storage_bytes: int = 0
    total_cost: float = 0.0
    cost_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class Analytics(BaseModel):
    """Analytics data container."""
    period: str  # "day", "week", "month", "custom"
    start_date: datetime
    end_date: datetime
    call_metrics: Optional[CallMetrics] = None
    usage_metrics: Optional[UsageMetrics] = None
    top_agents: List[Dict[str, Any]] = field(default_factory=list)
    call_volume_by_hour: Dict[str, int] = field(default_factory=dict)
    call_volume_by_day: Dict[str, int] = field(default_factory=dict)
    sentiment_distribution: Dict[str, int] = field(default_factory=dict)
    top_intents: List[Dict[str, Any]] = field(default_factory=list)
    average_handle_time: float = 0.0
    first_call_resolution_rate: float = 0.0


@dataclass
class Usage(BaseModel):
    """Current usage and limits."""
    organization_id: str
    period_start: datetime
    period_end: datetime
    calls_used: int = 0
    calls_limit: Optional[int] = None
    minutes_used: float = 0.0
    minutes_limit: Optional[float] = None
    storage_used_bytes: int = 0
    storage_limit_bytes: Optional[int] = None
    agents_used: int = 0
    agents_limit: Optional[int] = None
    phone_numbers_used: int = 0
    phone_numbers_limit: Optional[int] = None
    current_spend: float = 0.0
    spend_limit: Optional[float] = None


# =============================================================================
# Organization & User Models
# =============================================================================

@dataclass
class Organization(BaseModel):
    """Organization/tenant."""
    id: str
    name: str
    slug: str
    plan: str = "free"
    status: str = "active"
    owner_id: str = ""
    member_count: int = 0
    settings: Dict[str, Any] = field(default_factory=dict)
    billing_email: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class User(BaseModel):
    """User account."""
    id: str
    email: str
    name: Optional[str] = None
    role: str = "member"  # owner, admin, member, viewer
    organization_id: Optional[str] = None
    avatar_url: Optional[str] = None
    phone: Optional[str] = None
    timezone: str = "UTC"
    email_verified: bool = False
    last_login_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIKey(BaseModel):
    """API key for authentication."""
    id: str
    name: str
    key_prefix: str  # First 8 characters of the key
    key_hash: Optional[str] = None  # Not returned by API
    permissions: List[str] = field(default_factory=list)
    rate_limit: Optional[int] = None
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    organization_id: Optional[str] = None
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None


# =============================================================================
# Billing Models
# =============================================================================

@dataclass
class Invoice(BaseModel):
    """Billing invoice."""
    id: str
    organization_id: str
    number: str
    status: str  # draft, open, paid, void, uncollectible
    amount: float
    currency: str = "USD"
    period_start: datetime
    period_end: datetime
    due_date: Optional[datetime] = None
    paid_at: Optional[datetime] = None
    pdf_url: Optional[str] = None
    line_items: List[Dict[str, Any]] = field(default_factory=list)
    created_at: Optional[datetime] = None


@dataclass
class Subscription(BaseModel):
    """Subscription plan."""
    id: str
    organization_id: str
    plan_id: str
    plan_name: str
    status: str  # active, canceled, past_due, trialing
    current_period_start: datetime
    current_period_end: datetime
    cancel_at_period_end: bool = False
    canceled_at: Optional[datetime] = None
    trial_end: Optional[datetime] = None
    monthly_price: float = 0.0
    currency: str = "USD"
    created_at: Optional[datetime] = None


@dataclass
class PaymentMethod(BaseModel):
    """Payment method."""
    id: str
    type: str  # card, bank_account
    is_default: bool = False
    card_brand: Optional[str] = None
    card_last4: Optional[str] = None
    card_exp_month: Optional[int] = None
    card_exp_year: Optional[int] = None
    billing_address: Dict[str, str] = field(default_factory=dict)
    created_at: Optional[datetime] = None
