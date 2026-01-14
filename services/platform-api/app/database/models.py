"""Database models for Platform API."""

import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import (
    Column,
    String,
    Text,
    Boolean,
    Integer,
    Float,
    DateTime,
    ForeignKey,
    JSON,
    Enum,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class AgentStatus(str, PyEnum):
    """Agent status."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class CallStatus(str, PyEnum):
    """Call status."""
    QUEUED = "queued"
    INITIATED = "initiated"
    RINGING = "ringing"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NO_ANSWER = "no_answer"
    BUSY = "busy"
    CANCELLED = "cancelled"


class CallDirection(str, PyEnum):
    """Call direction."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class User(Base):
    """User model."""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    agents = relationship("Agent", back_populates="owner")
    api_keys = relationship("APIKey", back_populates="user")


class APIKey(Base):
    """API Key model."""
    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    key_hash = Column(String(255), nullable=False, unique=True, index=True)
    name = Column(String(255), nullable=False)
    scopes = Column(JSON, default=list)  # API key scopes/permissions
    is_active = Column(Boolean, default=True)
    last_used_at = Column(DateTime(timezone=True))
    revoked_at = Column(DateTime(timezone=True))  # When the key was revoked

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True))

    # Relationships
    user = relationship("User", back_populates="api_keys")


class Agent(Base):
    """Voice agent model."""
    __tablename__ = "agents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # Basic info
    name = Column(String(255), nullable=False)
    description = Column(Text)
    status = Column(Enum(AgentStatus), default=AgentStatus.DRAFT)

    # Voice settings
    voice_id = Column(String(100))  # ElevenLabs voice ID
    voice_name = Column(String(100))
    language = Column(String(10), default="en-US")

    # Behavior
    system_prompt = Column(Text)
    greeting_message = Column(Text)
    fallback_message = Column(Text)
    goodbye_message = Column(Text)

    # LLM settings
    llm_provider = Column(String(50), default="openai")
    llm_model = Column(String(100), default="gpt-4o-mini")
    temperature = Column(Float, default=0.7)
    max_tokens = Column(Integer, default=500)

    # Phone settings
    phone_number = Column(String(20))
    forward_number = Column(String(20))  # For transfers

    # Configuration
    tools = Column(JSON, default=list)  # Available functions
    settings = Column(JSON, default=dict)  # Additional settings

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    owner = relationship("User", back_populates="agents")
    calls = relationship("Call", back_populates="agent")
    knowledge_bases = relationship("KnowledgeBase", back_populates="agent")

    # Indexes
    __table_args__ = (
        Index("ix_agents_owner_status", "owner_id", "status"),
    )


class Call(Base):
    """Call record model."""
    __tablename__ = "calls"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=False)

    # Call identifiers
    session_id = Column(String(100), unique=True, index=True)
    external_id = Column(String(100))  # External provider ID (legacy)
    twilio_call_sid = Column(String(100), index=True)  # Twilio Call SID

    # Call info
    direction = Column(Enum(CallDirection), nullable=False)
    status = Column(Enum(CallStatus), default=CallStatus.INITIATED)
    from_number = Column(String(20))
    to_number = Column(String(20))

    # Timing
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    answered_at = Column(DateTime(timezone=True))
    ended_at = Column(DateTime(timezone=True))
    duration_seconds = Column(Integer, default=0)

    # Conversation
    transcript = Column(Text)
    summary = Column(Text)
    sentiment = Column(String(20))  # positive, neutral, negative

    # Outcomes
    outcome = Column(String(100))  # e.g., "appointment_booked", "transferred"
    outcome_details = Column(JSON)

    # Recording
    recording_url = Column(Text)
    recording_duration = Column(Integer)

    # Metadata
    metadata = Column(JSON, default=dict)

    # Relationships
    agent = relationship("Agent", back_populates="calls")
    logs = relationship("CallLog", back_populates="call")

    # Indexes
    __table_args__ = (
        Index("ix_calls_agent_started", "agent_id", "started_at"),
        Index("ix_calls_status", "status"),
    )


class CallLog(Base):
    """Call log entry for conversation turns and events."""
    __tablename__ = "call_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id = Column(UUID(as_uuid=True), ForeignKey("calls.id"), nullable=False)

    # Event info
    event_type = Column(String(50), nullable=False, index=True)  # speech, action, system, etc.
    speaker = Column(String(50))  # user, agent, system

    # Turn info (for conversation entries)
    turn_number = Column(Integer)
    role = Column(String(20))  # user, assistant, system (legacy)
    content = Column(Text)

    # Timing
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    duration_ms = Column(Integer)

    # Analysis
    intent = Column(String(100))
    entities = Column(JSON)
    confidence = Column(Float)

    # Function calls
    function_call = Column(JSON)
    function_result = Column(JSON)

    # Additional metadata
    metadata = Column(JSON, default=dict)

    # Relationships
    call = relationship("Call", back_populates="logs")

    # Indexes
    __table_args__ = (
        Index("ix_call_logs_call_turn", "call_id", "turn_number"),
        Index("ix_call_logs_event_type", "call_id", "event_type"),
    )


class KnowledgeBase(Base):
    """Knowledge base for agent RAG."""
    __tablename__ = "knowledge_bases"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=False)

    # Info
    name = Column(String(255), nullable=False)
    description = Column(Text)
    source_type = Column(String(50))  # url, file, text
    source_url = Column(Text)

    # Content
    content = Column(Text)
    chunk_count = Column(Integer, default=0)

    # Status
    is_indexed = Column(Boolean, default=False)
    indexed_at = Column(DateTime(timezone=True))

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    agent = relationship("Agent", back_populates="knowledge_bases")


class UsageRecord(Base):
    """Usage tracking for billing and analytics."""
    __tablename__ = "usage_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # Usage type
    usage_type = Column(String(50), nullable=False)  # api_call, llm_tokens, tts_chars, asr_minutes, storage_mb

    # Usage amount
    amount = Column(Integer, nullable=False, default=0)
    unit = Column(String(20), nullable=False)  # count, tokens, chars, minutes, mb

    # Context
    resource_type = Column(String(50))  # call, agent, knowledge_base
    resource_id = Column(UUID(as_uuid=True))

    # Details
    details = Column(JSON, default=dict)

    # Timestamp
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())

    # Indexes
    __table_args__ = (
        Index("ix_usage_user_type_date", "user_id", "usage_type", "recorded_at"),
        Index("ix_usage_recorded_at", "recorded_at"),
    )


class AuditEventModel(Base):
    """Audit event storage for compliance and security."""
    __tablename__ = "audit_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_id = Column(String(36), unique=True, nullable=False, index=True)

    # Event classification
    event_type = Column(String(100), nullable=False, index=True)
    severity = Column(String(20), nullable=False)

    # Actor and action
    actor_id = Column(String(255), index=True)
    action = Column(Text, nullable=False)
    outcome = Column(String(20), nullable=False)  # success, failure, error

    # Resource
    resource_type = Column(String(100))
    resource_id = Column(String(255))

    # Context
    request_id = Column(String(100))
    session_id = Column(String(100))
    tenant_id = Column(String(100), index=True)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    correlation_id = Column(String(100))

    # Details
    details = Column(JSON, default=dict)

    # Hash chain for tamper detection
    previous_hash = Column(String(64))
    event_hash = Column(String(64), nullable=False)

    # Timestamp
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Indexes
    __table_args__ = (
        Index("ix_audit_type_timestamp", "event_type", "timestamp"),
        Index("ix_audit_actor_timestamp", "actor_id", "timestamp"),
        Index("ix_audit_resource", "resource_type", "resource_id"),
    )


class ResponseTimeLog(Base):
    """Track response times for analytics."""
    __tablename__ = "response_time_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id = Column(UUID(as_uuid=True), ForeignKey("calls.id"), nullable=False)

    # Timing breakdown
    asr_latency_ms = Column(Integer)  # Speech-to-text time
    llm_latency_ms = Column(Integer)  # LLM response time
    tts_latency_ms = Column(Integer)  # Text-to-speech time
    total_latency_ms = Column(Integer)  # Total round-trip

    # Turn info
    turn_number = Column(Integer)

    # Timestamp
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())

    # Indexes
    __table_args__ = (
        Index("ix_response_call_turn", "call_id", "turn_number"),
        Index("ix_response_recorded", "recorded_at"),
    )


class QueueWaitTime(Base):
    """Track call queue wait times."""
    __tablename__ = "queue_wait_times"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id = Column(UUID(as_uuid=True), ForeignKey("calls.id"), nullable=False)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=False)

    # Wait time in seconds
    wait_time_seconds = Column(Integer, nullable=False)

    # Queue position when entered
    initial_position = Column(Integer)

    # Timestamps
    entered_queue_at = Column(DateTime(timezone=True), nullable=False)
    exited_queue_at = Column(DateTime(timezone=True))

    # Outcome
    outcome = Column(String(50))  # answered, abandoned, transferred

    # Index
    __table_args__ = (
        Index("ix_queue_agent_entered", "agent_id", "entered_queue_at"),
    )
