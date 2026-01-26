"""
Database Models

SQLAlchemy ORM models for all platform entities.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from decimal import Decimal
from sqlalchemy import (
    Column,
    DateTime,
    String,
    Boolean,
    Text,
    Integer,
    Float,
    Numeric,
    ForeignKey,
    Index,
    Enum,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB

# Use JSONB for PostgreSQL, fall back to JSON for other databases
# JSONB provides better query performance and supports GIN indexing
JSON = JSONB  # Override to use JSONB by default for PostgreSQL compatibility
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
    slug: Mapped[str] = mapped_column(String(100), nullable=False)

    # Configuration
    system_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    first_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # LLM Configuration
    llm_provider: Mapped[str] = mapped_column(String(50), default="openai")
    llm_model: Mapped[str] = mapped_column(String(100), default="gpt-4")
    llm_temperature: Mapped[float] = mapped_column(Float, default=0.7)
    llm_max_tokens: Mapped[int] = mapped_column(Integer, default=150)

    # Voice Configuration
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

    # Metadata
    metadata_json: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
    tags: Mapped[Optional[List]] = mapped_column(JSON, nullable=True)

    # Relationships
    organization = relationship("Organization", back_populates="agents")
    versions = relationship("AgentVersion", back_populates="agent")
    conversations = relationship("Conversation", back_populates="agent")
    calls = relationship("Call", back_populates="agent")
    voice_config = relationship("VoiceConfigurationModel", back_populates="agent", uselist=False)

    __table_args__ = (
        UniqueConstraint("organization_id", "slug", name="uq_agent_org_slug"),
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
    organization_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )

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

    organization_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )
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
        # Single column indexes
        Index("ix_conversations_organization_id", "organization_id"),
        Index("ix_conversations_agent_id", "agent_id"),
        Index("ix_conversations_call_id", "call_id"),
        Index("ix_conversations_customer_phone", "customer_phone"),
        Index("ix_conversations_status", "status"),
        Index("ix_conversations_started_at", "started_at"),
        # Composite indexes for common query patterns
        Index("ix_conversations_org_started", "organization_id", "started_at"),
        Index("ix_conversations_org_status", "organization_id", "status"),
        Index("ix_conversations_org_agent_started", "organization_id", "agent_id", "started_at"),
        # Partial index for active conversations
        Index(
            "ix_conversations_org_active",
            "organization_id",
            "started_at",
            postgresql_where=text("is_deleted = false AND status = 'active'"),
        ),
    )


class Message(Base, TimestampMixin, SoftDeleteMixin):
    """Conversation message model."""

    __tablename__ = "messages"

    conversation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
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
        # Composite index for message pagination (most common query pattern)
        Index("ix_messages_conv_created", "conversation_id", "created_at"),
        # Index for sentiment analysis queries
        Index("ix_messages_role_sentiment", "role", "sentiment"),
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

    # Cost - Using Numeric for precise monetary calculations
    cost_amount: Mapped[Decimal] = mapped_column(
        Numeric(12, 4),  # Supports up to $99,999,999.9999
        default=Decimal("0.0000"),
    )
    cost_currency: Mapped[str] = mapped_column(String(10), default="USD")

    # Metadata
    metadata_json: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    # Relationships
    agent = relationship("Agent", back_populates="calls")
    events = relationship("CallEvent", back_populates="call", order_by="CallEvent.created_at")

    __table_args__ = (
        # Single column indexes
        Index("ix_calls_organization_id", "organization_id"),
        Index("ix_calls_agent_id", "agent_id"),
        Index("ix_calls_external_call_id", "external_call_id"),
        Index("ix_calls_status", "status"),
        Index("ix_calls_from_number", "from_number"),
        Index("ix_calls_to_number", "to_number"),
        Index("ix_calls_initiated_at", "initiated_at"),
        # Composite indexes for common query patterns (dashboards, filtering)
        Index("ix_calls_org_status_initiated", "organization_id", "status", "initiated_at"),
        Index("ix_calls_org_agent_initiated", "organization_id", "agent_id", "initiated_at"),
        # Partial index for active calls (soft delete filtering)
        Index(
            "ix_calls_org_active",
            "organization_id",
            "initiated_at",
            postgresql_where=text("is_deleted = false"),
        ),
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

    # Cost - Using Numeric for precise monetary calculations
    unit_cost: Mapped[Decimal] = mapped_column(
        Numeric(12, 6),  # Higher precision for unit costs (e.g., $0.000001/token)
        default=Decimal("0.000000"),
    )
    total_cost: Mapped[Decimal] = mapped_column(
        Numeric(12, 4),  # Supports up to $99,999,999.9999
        default=Decimal("0.0000"),
    )
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
]
