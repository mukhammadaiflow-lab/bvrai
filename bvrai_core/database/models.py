"""
Database Models

SQLAlchemy ORM models for all platform entities.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    String,
    Boolean,
    Text,
    Integer,
    Float,
    ForeignKey,
    JSON,
    Index,
    Enum,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin, SoftDeleteMixin, AuditMixin


# =============================================================================
# Organization Models
# =============================================================================


class Organization(Base, TimestampMixin, SoftDeleteMixin):
    """Organization model."""

    __tablename__ = "organizations"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Contact
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    website: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Billing
    billing_email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    plan: Mapped[str] = mapped_column(String(50), default="free")
    stripe_customer_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)

    # Metadata
    metadata_json: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    # Relationships
    users = relationship("User", back_populates="organization")
    agents = relationship("Agent", back_populates="organization")
    api_keys = relationship("APIKey", back_populates="organization")
    settings = relationship("OrganizationSettings", back_populates="organization", uselist=False)

    __table_args__ = (
        Index("ix_organizations_slug", "slug"),
        Index("ix_organizations_is_active", "is_active"),
    )


class OrganizationSettings(Base, TimestampMixin):
    """Organization settings model."""

    __tablename__ = "organization_settings"

    organization_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    # Default settings
    default_language: Mapped[str] = mapped_column(String(10), default="en-US")
    default_timezone: Mapped[str] = mapped_column(String(50), default="UTC")

    # Voice settings
    default_stt_provider: Mapped[str] = mapped_column(String(50), default="deepgram")
    default_tts_provider: Mapped[str] = mapped_column(String(50), default="elevenlabs")
    default_llm_provider: Mapped[str] = mapped_column(String(50), default="openai")

    # Limits
    max_agents: Mapped[int] = mapped_column(Integer, default=10)
    max_concurrent_calls: Mapped[int] = mapped_column(Integer, default=100)
    monthly_minutes_limit: Mapped[int] = mapped_column(Integer, default=1000)

    # Features
    features_enabled: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    # Webhooks
    webhook_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    webhook_secret: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Relationships
    organization = relationship("Organization", back_populates="settings")


# =============================================================================
# User Models
# =============================================================================


class User(Base, TimestampMixin, SoftDeleteMixin):
    """User model."""

    __tablename__ = "users"

    organization_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )

    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Profile
    first_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    last_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    avatar_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Role
    role: Mapped[str] = mapped_column(String(50), default="member")  # owner, admin, member

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Auth
    email_verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    two_factor_enabled: Mapped[bool] = mapped_column(Boolean, default=False)

    # Metadata
    preferences: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    # Relationships
    organization = relationship("Organization", back_populates="users")

    __table_args__ = (
        Index("ix_users_email", "email"),
        Index("ix_users_organization_id", "organization_id"),
    )

    @property
    def full_name(self) -> str:
        """Get full name."""
        parts = [self.first_name, self.last_name]
        return " ".join(p for p in parts if p) or self.email

    @property
    def name(self) -> str:
        """Get full name (alias for full_name)."""
        return self.full_name

    @name.setter
    def name(self, value: str) -> None:
        """Set name by splitting into first/last."""
        parts = value.strip().split(" ", 1)
        self.first_name = parts[0]
        self.last_name = parts[1] if len(parts) > 1 else None

    def set_password(self, password: str) -> None:
        """Hash and set the user's password."""
        import hashlib
        # Simple hash for MVP - use bcrypt/argon2 in production
        self.password_hash = hashlib.sha256(password.encode()).hexdigest()

    def verify_password(self, password: str) -> bool:
        """Verify the user's password."""
        import hashlib
        if not self.password_hash:
            return False
        return self.password_hash == hashlib.sha256(password.encode()).hexdigest()


class APIKey(Base, TimestampMixin, SoftDeleteMixin):
    """API key model."""

    __tablename__ = "api_keys"

    organization_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(String(100), nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(20), nullable=False)  # First 8 chars
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    # Permissions
    scopes: Mapped[Optional[List]] = mapped_column(JSON, nullable=True)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Usage
    request_count: Mapped[int] = mapped_column(Integer, default=0)

    # Created by
    created_by_user_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    # Relationships
    organization = relationship("Organization", back_populates="api_keys")

    __table_args__ = (
        Index("ix_api_keys_key_prefix", "key_prefix"),
        Index("ix_api_keys_organization_id", "organization_id"),
    )


# =============================================================================
# Agent Models
# =============================================================================


class Agent(Base, TimestampMixin, SoftDeleteMixin, AuditMixin):
    """AI Agent model."""

    __tablename__ = "agents"

    organization_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Basic info
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    slug: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    industry: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Configuration
    system_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    first_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # JSON Configuration (flexible storage)
    voice_config_json: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
    llm_config: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
    behavior_config: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
    transcription_config: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    # Knowledge bases and tools
    knowledge_base_ids: Mapped[Optional[List]] = mapped_column(JSON, nullable=True)
    functions: Mapped[Optional[List]] = mapped_column(JSON, nullable=True)

    # Legacy LLM Configuration (for backwards compatibility)
    llm_provider: Mapped[str] = mapped_column(String(50), default="openai")
    llm_model: Mapped[str] = mapped_column(String(100), default="gpt-4")
    llm_temperature: Mapped[float] = mapped_column(Float, default=0.7)
    llm_max_tokens: Mapped[int] = mapped_column(Integer, default=150)

    # Legacy Voice Configuration
    voice_config_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_published: Mapped[bool] = mapped_column(Boolean, default=False)

    # Version
    current_version: Mapped[int] = mapped_column(Integer, default=1)

    # Phone number
    phone_number: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Usage stats
    total_calls: Mapped[int] = mapped_column(Integer, default=0)
    total_minutes: Mapped[float] = mapped_column(Float, default=0.0)

    # Metadata (using 'extra_data' to avoid SQLAlchemy reserved 'metadata')
    extra_data: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
    metadata_json: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)  # Legacy
    tags: Mapped[Optional[List]] = mapped_column(JSON, nullable=True)

    # Relationships
    organization = relationship("Organization", back_populates="agents")
    versions = relationship("AgentVersion", back_populates="agent")
    conversations = relationship("Conversation", back_populates="agent")
    calls = relationship("Call", back_populates="agent")
    voice_config = relationship("VoiceConfigurationModel", back_populates="agent", uselist=False)

    __table_args__ = (
        Index("ix_agents_organization_id", "organization_id"),
        Index("ix_agents_is_active", "is_active"),
    )


class AgentVersion(Base, TimestampMixin):
    """Agent version history model."""

    __tablename__ = "agent_versions"

    agent_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
    )

    version_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Snapshot of configuration
    system_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    first_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    llm_provider: Mapped[str] = mapped_column(String(50), nullable=False)
    llm_model: Mapped[str] = mapped_column(String(100), nullable=False)
    llm_temperature: Mapped[float] = mapped_column(Float, nullable=False)

    # Full config snapshot
    config_snapshot: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    # Metadata
    change_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_by_user_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    # Relationships
    agent = relationship("Agent", back_populates="versions")

    __table_args__ = (
        UniqueConstraint("agent_id", "version_number", name="uq_agent_version"),
        Index("ix_agent_versions_agent_id", "agent_id"),
    )


# =============================================================================
# Voice Configuration Model
# =============================================================================


class VoiceConfigurationModel(Base, TimestampMixin):
    """Voice configuration model (database representation)."""

    __tablename__ = "voice_configurations"

    agent_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    organization_id: Mapped[str] = mapped_column(String(36), nullable=False)

    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # STT Configuration
    stt_provider: Mapped[str] = mapped_column(String(50), default="deepgram")
    stt_model_id: Mapped[str] = mapped_column(String(100), default="nova-2")
    stt_language: Mapped[str] = mapped_column(String(10), default="en-US")

    # TTS Configuration
    tts_provider: Mapped[str] = mapped_column(String(50), default="elevenlabs")
    tts_model_id: Mapped[str] = mapped_column(String(100), default="eleven_turbo_v2_5")
    tts_voice_id: Mapped[str] = mapped_column(String(255), default="")
    tts_custom_voice_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Voice settings
    voice_speed: Mapped[float] = mapped_column(Float, default=1.0)
    voice_pitch: Mapped[float] = mapped_column(Float, default=1.0)
    voice_stability: Mapped[float] = mapped_column(Float, default=0.5)
    voice_similarity_boost: Mapped[float] = mapped_column(Float, default=0.75)

    # VAD settings
    vad_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    vad_threshold: Mapped[float] = mapped_column(Float, default=0.5)

    # Turn-taking
    allow_interruption: Mapped[bool] = mapped_column(Boolean, default=True)
    turn_end_silence_ms: Mapped[int] = mapped_column(Integer, default=700)

    # Features
    backchanneling_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    filler_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    noise_suppression: Mapped[bool] = mapped_column(Boolean, default=True)

    # Full configuration JSON
    full_config: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    # Relationships
    agent = relationship("Agent", back_populates="voice_config")


# =============================================================================
# Conversation Models
# =============================================================================


class Conversation(Base, TimestampMixin, SoftDeleteMixin):
    """Conversation model."""

    __tablename__ = "conversations"

    organization_id: Mapped[str] = mapped_column(String(36), nullable=False)
    agent_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("agents.id", ondelete="SET NULL"),
        nullable=True,
    )
    call_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    # Customer info
    customer_phone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    customer_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    customer_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Status
    status: Mapped[str] = mapped_column(String(50), default="active")  # active, completed, abandoned

    # Timing
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)

    # Stats
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    turn_count: Mapped[int] = mapped_column(Integer, default=0)

    # Analysis
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    topics: Mapped[Optional[List]] = mapped_column(JSON, nullable=True)
    entities: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    # Outcome
    outcome: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    resolution: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Metadata
    metadata_json: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    # Relationships
    agent = relationship("Agent", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", order_by="Message.created_at")

    __table_args__ = (
        Index("ix_conversations_organization_id", "organization_id"),
        Index("ix_conversations_agent_id", "agent_id"),
        Index("ix_conversations_call_id", "call_id"),
        Index("ix_conversations_customer_phone", "customer_phone"),
        Index("ix_conversations_status", "status"),
        Index("ix_conversations_started_at", "started_at"),
    )


class Message(Base, TimestampMixin, SoftDeleteMixin):
    """Conversation message model."""

    __tablename__ = "messages"

    conversation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Message content
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user, assistant, system
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Audio
    audio_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    audio_duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Timing
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Analysis
    sentiment: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    intent: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    entities: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    # Token counts
    input_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Metadata
    metadata_json: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

    __table_args__ = (
        Index("ix_messages_conversation_id", "conversation_id"),
        Index("ix_messages_role", "role"),
        Index("ix_messages_created_at", "created_at"),
    )


# =============================================================================
# Call Models
# =============================================================================


class Call(Base, TimestampMixin, SoftDeleteMixin):
    """Call model."""

    __tablename__ = "calls"

    organization_id: Mapped[str] = mapped_column(String(36), nullable=False)
    agent_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("agents.id", ondelete="SET NULL"),
        nullable=True,
    )
    conversation_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Call identifiers
    external_call_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    telephony_provider: Mapped[str] = mapped_column(String(50), default="twilio")

    # Direction
    direction: Mapped[str] = mapped_column(String(20), default="inbound")  # inbound, outbound

    # Phone numbers
    from_number: Mapped[str] = mapped_column(String(50), nullable=False)
    to_number: Mapped[str] = mapped_column(String(50), nullable=False)

    # Status
    status: Mapped[str] = mapped_column(String(50), default="initiated")
    # initiated, ringing, in_progress, completed, failed, no_answer, busy, cancelled

    # Timing
    initiated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    answered_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    ring_duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)

    # Recording
    recording_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    transcript_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Quality
    audio_quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # End reason
    end_reason: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    hangup_source: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # caller, agent, system

    # Transfer
    transferred: Mapped[bool] = mapped_column(Boolean, default=False)
    transferred_to: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    transfer_reason: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Cost
    cost_amount: Mapped[float] = mapped_column(Float, default=0.0)
    cost_currency: Mapped[str] = mapped_column(String(10), default="USD")

    # Metadata
    metadata_json: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    # Relationships
    agent = relationship("Agent", back_populates="calls")
    events = relationship("CallEvent", back_populates="call", order_by="CallEvent.created_at")

    __table_args__ = (
        Index("ix_calls_organization_id", "organization_id"),
        Index("ix_calls_agent_id", "agent_id"),
        Index("ix_calls_external_call_id", "external_call_id"),
        Index("ix_calls_status", "status"),
        Index("ix_calls_from_number", "from_number"),
        Index("ix_calls_to_number", "to_number"),
        Index("ix_calls_initiated_at", "initiated_at"),
    )


class CallEvent(Base, TimestampMixin):
    """Call event model for tracking call lifecycle."""

    __tablename__ = "call_events"

    call_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("calls.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Event details
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # call_initiated, call_ringing, call_answered, call_ended,
    # speech_started, speech_ended, agent_response, transfer_initiated, etc.

    event_data: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    # Timing
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    relative_time_ms: Mapped[int] = mapped_column(Integer, default=0)  # ms from call start

    # Relationships
    call = relationship("Call", back_populates="events")

    __table_args__ = (
        Index("ix_call_events_call_id", "call_id"),
        Index("ix_call_events_event_type", "event_type"),
        Index("ix_call_events_timestamp", "timestamp"),
    )


# =============================================================================
# Analytics Models
# =============================================================================


class AnalyticsEvent(Base, TimestampMixin):
    """Analytics event model."""

    __tablename__ = "analytics_events"

    organization_id: Mapped[str] = mapped_column(String(36), nullable=False)

    # Event details
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    event_name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Context
    agent_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    call_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    conversation_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    # Event data
    properties: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    # Source
    source: Mapped[str] = mapped_column(String(50), default="system")
    ip_address: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    __table_args__ = (
        Index("ix_analytics_events_organization_id", "organization_id"),
        Index("ix_analytics_events_event_type", "event_type"),
        Index("ix_analytics_events_agent_id", "agent_id"),
        Index("ix_analytics_events_created_at", "created_at"),
    )


class UsageRecord(Base, TimestampMixin):
    """Usage record for billing and quotas."""

    __tablename__ = "usage_records"

    organization_id: Mapped[str] = mapped_column(String(36), nullable=False)

    # Usage type
    usage_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # stt_minutes, tts_characters, llm_tokens, calls, storage_gb

    # Period
    period_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    period_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Quantities
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(20), nullable=False)

    # Cost
    unit_cost: Mapped[float] = mapped_column(Float, default=0.0)
    total_cost: Mapped[float] = mapped_column(Float, default=0.0)
    currency: Mapped[str] = mapped_column(String(10), default="USD")

    # Context
    agent_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    # Billing
    invoiced: Mapped[bool] = mapped_column(Boolean, default=False)
    invoice_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    __table_args__ = (
        Index("ix_usage_records_organization_id", "organization_id"),
        Index("ix_usage_records_usage_type", "usage_type"),
        Index("ix_usage_records_period_start", "period_start"),
    )


# =============================================================================
# Phone Number Models
# =============================================================================


class PhoneNumber(Base, TimestampMixin, SoftDeleteMixin):
    """Phone number model for telephony management."""

    __tablename__ = "phone_numbers"

    organization_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Phone number details
    number: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)  # E.164 format
    friendly_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Provider info
    provider: Mapped[str] = mapped_column(String(50), default="twilio")  # twilio, vonage, bandwidth
    provider_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Type and capabilities
    number_type: Mapped[str] = mapped_column(String(20), default="local")  # local, toll_free, mobile
    country_code: Mapped[str] = mapped_column(String(5), default="US")

    # Capabilities
    voice_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    sms_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    mms_enabled: Mapped[bool] = mapped_column(Boolean, default=False)

    # Assignment
    agent_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("agents.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Configuration
    webhook_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    fallback_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    status_callback_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Status
    status: Mapped[str] = mapped_column(String(20), default="active")  # active, inactive, pending, released

    # Billing
    monthly_cost: Mapped[float] = mapped_column(Float, default=0.0)
    currency: Mapped[str] = mapped_column(String(10), default="USD")

    # Metadata
    extra_data: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    # Relationships
    organization = relationship("Organization")
    agent = relationship("Agent")

    __table_args__ = (
        Index("ix_phone_numbers_organization_id", "organization_id"),
        Index("ix_phone_numbers_number", "number"),
        Index("ix_phone_numbers_agent_id", "agent_id"),
        Index("ix_phone_numbers_status", "status"),
    )


# =============================================================================
# Webhook Models
# =============================================================================


class Webhook(Base, TimestampMixin, SoftDeleteMixin):
    """Webhook configuration model."""

    __tablename__ = "webhooks"

    organization_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Basic info
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # URL and authentication
    url: Mapped[str] = mapped_column(String(500), nullable=False)
    secret: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # For signing requests

    # Authentication method
    auth_type: Mapped[str] = mapped_column(String(20), default="none")  # none, basic, bearer, hmac
    auth_value: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)  # Encrypted

    # Events to subscribe to
    events: Mapped[List] = mapped_column(JSON, default=list)  # List of event types
    # Examples: call.started, call.ended, agent.updated, campaign.completed

    # Filtering
    agent_ids: Mapped[Optional[List]] = mapped_column(JSON, nullable=True)  # Filter by specific agents

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Retry configuration
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    retry_delay_seconds: Mapped[int] = mapped_column(Integer, default=60)
    timeout_seconds: Mapped[int] = mapped_column(Integer, default=30)

    # Statistics
    total_deliveries: Mapped[int] = mapped_column(Integer, default=0)
    successful_deliveries: Mapped[int] = mapped_column(Integer, default=0)
    failed_deliveries: Mapped[int] = mapped_column(Integer, default=0)
    last_triggered_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_success_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_failure_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Metadata
    extra_data: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    # Relationships
    organization = relationship("Organization")
    deliveries = relationship("WebhookDelivery", back_populates="webhook", order_by="WebhookDelivery.created_at.desc()")

    __table_args__ = (
        Index("ix_webhooks_organization_id", "organization_id"),
        Index("ix_webhooks_is_active", "is_active"),
    )


class WebhookDelivery(Base, TimestampMixin):
    """Webhook delivery attempt record."""

    __tablename__ = "webhook_deliveries"

    webhook_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("webhooks.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Event info
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    event_id: Mapped[str] = mapped_column(String(36), nullable=False)  # Unique event identifier

    # Request details
    request_url: Mapped[str] = mapped_column(String(500), nullable=False)
    request_headers: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
    request_body: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    # Response details
    response_status: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    response_headers: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
    response_body: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timing
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Status
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, success, failed, retrying
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Retry info
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)
    next_retry_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    webhook = relationship("Webhook", back_populates="deliveries")

    __table_args__ = (
        Index("ix_webhook_deliveries_webhook_id", "webhook_id"),
        Index("ix_webhook_deliveries_event_type", "event_type"),
        Index("ix_webhook_deliveries_status", "status"),
        Index("ix_webhook_deliveries_created_at", "created_at"),
    )


# =============================================================================
# Knowledge Base Models
# =============================================================================


class KnowledgeBase(Base, TimestampMixin, SoftDeleteMixin):
    """Knowledge base model for RAG functionality."""

    __tablename__ = "knowledge_bases"

    organization_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Basic info
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Configuration
    embedding_model: Mapped[str] = mapped_column(String(100), default="text-embedding-3-small")
    embedding_provider: Mapped[str] = mapped_column(String(50), default="openai")
    chunk_size: Mapped[int] = mapped_column(Integer, default=1000)
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=200)

    # Vector store
    vector_store: Mapped[str] = mapped_column(String(50), default="qdrant")  # qdrant, pinecone, weaviate
    vector_collection: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Status
    status: Mapped[str] = mapped_column(String(20), default="active")  # active, processing, error

    # Statistics
    document_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    last_synced_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Metadata
    extra_data: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    # Relationships
    organization = relationship("Organization")
    documents = relationship("Document", back_populates="knowledge_base", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_knowledge_bases_organization_id", "organization_id"),
        Index("ix_knowledge_bases_status", "status"),
    )


class Document(Base, TimestampMixin, SoftDeleteMixin):
    """Document within a knowledge base."""

    __tablename__ = "documents"

    knowledge_base_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    organization_id: Mapped[str] = mapped_column(String(36), nullable=False)

    # Basic info
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Document type
    doc_type: Mapped[str] = mapped_column(String(50), default="text")  # text, pdf, url, faq, csv

    # Content
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # For text/FAQ types
    source_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)  # For URL types
    file_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)  # For file uploads
    file_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    mime_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Processing status
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, processing, completed, failed
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Processing stats
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    token_count: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Metadata
    extra_data: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    # Relationships
    knowledge_base = relationship("KnowledgeBase", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_documents_knowledge_base_id", "knowledge_base_id"),
        Index("ix_documents_organization_id", "organization_id"),
        Index("ix_documents_status", "status"),
        Index("ix_documents_doc_type", "doc_type"),
    )


class DocumentChunk(Base, TimestampMixin):
    """Document chunk for vector embedding."""

    __tablename__ = "document_chunks"

    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    knowledge_base_id: Mapped[str] = mapped_column(String(36), nullable=False)

    # Chunk content
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Position
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_char: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    end_char: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Token info
    token_count: Mapped[int] = mapped_column(Integer, default=0)

    # Vector info
    vector_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)  # ID in vector store
    embedding_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Metadata for retrieval
    chunk_metadata: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    # Relationships
    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("ix_document_chunks_document_id", "document_id"),
        Index("ix_document_chunks_knowledge_base_id", "knowledge_base_id"),
        Index("ix_document_chunks_vector_id", "vector_id"),
    )


# =============================================================================
# Campaign Models
# =============================================================================


class Campaign(Base, TimestampMixin, SoftDeleteMixin):
    """Outbound calling campaign model."""

    __tablename__ = "campaigns"

    organization_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Basic info
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Agent and phone number
    agent_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("agents.id", ondelete="SET NULL"),
        nullable=True,
    )
    phone_number_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("phone_numbers.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Schedule configuration
    schedule_config: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
    # Contains: start_time, end_time, timezone, daily_start_hour, daily_end_hour,
    #           days_of_week, max_concurrent_calls, calls_per_minute

    # Retry configuration
    retry_config: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
    # Contains: max_attempts, retry_delay_minutes, retry_on_busy, retry_on_no_answer, retry_on_voicemail

    # Status
    status: Mapped[str] = mapped_column(String(20), default="draft")
    # draft, scheduled, running, paused, completed, canceled

    # Statistics
    total_contacts: Mapped[int] = mapped_column(Integer, default=0)
    calls_completed: Mapped[int] = mapped_column(Integer, default=0)
    calls_successful: Mapped[int] = mapped_column(Integer, default=0)
    calls_failed: Mapped[int] = mapped_column(Integer, default=0)
    calls_pending: Mapped[int] = mapped_column(Integer, default=0)
    calls_in_progress: Mapped[int] = mapped_column(Integer, default=0)

    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    paused_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Cost tracking
    total_cost: Mapped[float] = mapped_column(Float, default=0.0)
    total_minutes: Mapped[float] = mapped_column(Float, default=0.0)

    # Metadata
    extra_data: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    # Relationships
    organization = relationship("Organization")
    agent = relationship("Agent")
    phone_number = relationship("PhoneNumber")
    contacts = relationship("CampaignContact", back_populates="campaign", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_campaigns_organization_id", "organization_id"),
        Index("ix_campaigns_agent_id", "agent_id"),
        Index("ix_campaigns_status", "status"),
        Index("ix_campaigns_created_at", "created_at"),
    )


class CampaignContact(Base, TimestampMixin):
    """Contact within a campaign."""

    __tablename__ = "campaign_contacts"

    campaign_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("campaigns.id", ondelete="CASCADE"),
        nullable=False,
    )
    organization_id: Mapped[str] = mapped_column(String(36), nullable=False)

    # Contact info
    phone_number: Mapped[str] = mapped_column(String(50), nullable=False)  # E.164 format
    first_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    last_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Custom context for the call
    context: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    # Call status
    status: Mapped[str] = mapped_column(String(20), default="pending")
    # pending, queued, calling, completed, failed, skipped

    # Call details
    call_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    call_outcome: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    # answered, voicemail, no_answer, busy, failed

    # Attempt tracking
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    last_attempt_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    next_attempt_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Result
    call_duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    call_cost: Mapped[float] = mapped_column(Float, default=0.0)

    # Notes
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Metadata
    extra_data: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    # Relationships
    campaign = relationship("Campaign", back_populates="contacts")

    __table_args__ = (
        Index("ix_campaign_contacts_campaign_id", "campaign_id"),
        Index("ix_campaign_contacts_organization_id", "organization_id"),
        Index("ix_campaign_contacts_status", "status"),
        Index("ix_campaign_contacts_phone_number", "phone_number"),
        UniqueConstraint("campaign_id", "phone_number", name="uq_campaign_contact_phone"),
    )


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Organization
    "Organization",
    "OrganizationSettings",
    # User/API
    "User",
    "APIKey",
    # Agent
    "Agent",
    "AgentVersion",
    # Voice Config
    "VoiceConfigurationModel",
    # Conversation
    "Conversation",
    "Message",
    # Call
    "Call",
    "CallEvent",
    # Analytics
    "AnalyticsEvent",
    "UsageRecord",
    # Phone Numbers
    "PhoneNumber",
    # Webhooks
    "Webhook",
    "WebhookDelivery",
    # Knowledge Base
    "KnowledgeBase",
    "Document",
    "DocumentChunk",
    # Campaigns
    "Campaign",
    "CampaignContact",
]
