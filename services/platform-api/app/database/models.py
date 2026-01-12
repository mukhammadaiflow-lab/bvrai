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
    INITIATED = "initiated"
    RINGING = "ringing"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NO_ANSWER = "no_answer"
    BUSY = "busy"


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
    is_active = Column(Boolean, default=True)
    last_used_at = Column(DateTime(timezone=True))

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
    external_id = Column(String(100))  # Twilio Call SID

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
    """Call log entry for conversation turns."""
    __tablename__ = "call_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id = Column(UUID(as_uuid=True), ForeignKey("calls.id"), nullable=False)

    # Turn info
    turn_number = Column(Integer, nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)

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

    # Relationships
    call = relationship("Call", back_populates="logs")

    # Indexes
    __table_args__ = (
        Index("ix_call_logs_call_turn", "call_id", "turn_number"),
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
