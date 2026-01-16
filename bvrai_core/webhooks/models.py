"""Webhook data models."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


class WebhookEventType(str, Enum):
    """Available webhook event types."""

    # Call events
    CALL_STARTED = "call.started"
    CALL_RINGING = "call.ringing"
    CALL_ANSWERED = "call.answered"
    CALL_ENDED = "call.ended"
    CALL_FAILED = "call.failed"

    # Transcription events
    TRANSCRIPTION_PARTIAL = "transcription.partial"
    TRANSCRIPTION_FINAL = "transcription.final"

    # Agent events
    AGENT_SPEECH_START = "agent.speech.start"
    AGENT_SPEECH_END = "agent.speech.end"
    AGENT_THINKING = "agent.thinking"
    AGENT_TOOL_CALL = "agent.tool_call"

    # Conversation events
    CONVERSATION_CREATED = "conversation.created"
    CONVERSATION_UPDATED = "conversation.updated"
    CONVERSATION_ENDED = "conversation.ended"

    # Campaign events
    CAMPAIGN_STARTED = "campaign.started"
    CAMPAIGN_COMPLETED = "campaign.completed"
    CAMPAIGN_PAUSED = "campaign.paused"
    CAMPAIGN_RESUMED = "campaign.resumed"
    CAMPAIGN_CANCELED = "campaign.canceled"

    # Phone number events
    PHONE_NUMBER_PURCHASED = "phone_number.purchased"
    PHONE_NUMBER_RELEASED = "phone_number.released"
    PHONE_NUMBER_CONFIGURED = "phone_number.configured"

    # Account events
    ACCOUNT_CREDITS_LOW = "account.credits_low"
    ACCOUNT_USAGE_LIMIT = "account.usage_limit"

    # Test event
    TEST_PING = "test.ping"

    @classmethod
    def matches(cls, event_type: str, pattern: str) -> bool:
        """Check if an event type matches a pattern (supports wildcards)."""
        if pattern == "*":
            return True
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return event_type.startswith(prefix)
        return event_type == pattern


class DeliveryStatus(str, Enum):
    """Webhook delivery status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


class WebhookStatus(str, Enum):
    """Webhook endpoint status."""
    ACTIVE = "active"
    DISABLED = "disabled"
    FAILED = "failed"  # Auto-disabled after too many failures


@dataclass
class Webhook:
    """Webhook endpoint configuration."""
    id: str
    organization_id: str
    url: str
    events: List[str]
    secret: str
    status: WebhookStatus = WebhookStatus.ACTIVE
    description: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    retry_enabled: bool = True
    max_retries: int = 3
    retry_delay_seconds: int = 60
    timeout_seconds: int = 30
    consecutive_failures: int = 0
    last_triggered_at: Optional[datetime] = None
    last_success_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_triggered_at": self.last_triggered_at.isoformat() if self.last_triggered_at else None,
            "last_success_at": self.last_success_at.isoformat() if self.last_success_at else None,
        }

    def should_receive_event(self, event_type: str) -> bool:
        """Check if this webhook should receive an event type."""
        if self.status != WebhookStatus.ACTIVE:
            return False

        for pattern in self.events:
            if WebhookEventType.matches(event_type, pattern):
                return True

        return False


@dataclass
class WebhookEvent:
    """A webhook event to be delivered."""
    id: str
    webhook_id: str
    organization_id: str
    event_type: str
    payload: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class DeliveryAttempt:
    """A single delivery attempt."""
    attempt_number: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    status_code: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None
    response_time_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class DeliveryLog:
    """Log of a webhook delivery (including retries)."""
    id: str
    webhook_id: str
    event_id: str
    event_type: str
    organization_id: str
    url: str
    status: DeliveryStatus
    payload: Dict[str, Any]
    attempts: List[DeliveryAttempt] = field(default_factory=list)
    next_retry_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    @property
    def total_attempts(self) -> int:
        """Get total number of attempts."""
        return len(self.attempts)

    @property
    def last_attempt(self) -> Optional[DeliveryAttempt]:
        """Get the most recent attempt."""
        return self.attempts[-1] if self.attempts else None

    @property
    def response_status(self) -> Optional[int]:
        """Get the last response status code."""
        if self.last_attempt:
            return self.last_attempt.status_code
        return None

    @property
    def response_time_ms(self) -> Optional[int]:
        """Get the last response time."""
        if self.last_attempt:
            return self.last_attempt.response_time_ms
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "webhook_id": self.webhook_id,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "organization_id": self.organization_id,
            "url": self.url,
            "status": self.status.value,
            "attempts": len(self.attempts),
            "response_status": self.response_status,
            "response_time_ms": self.response_time_ms,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "next_retry_at": self.next_retry_at.isoformat() if self.next_retry_at else None,
        }

    def to_dict_full(self) -> Dict[str, Any]:
        """Convert to dictionary with full details."""
        return {
            **self.to_dict(),
            "payload": self.payload,
            "attempts": [a.to_dict() for a in self.attempts],
        }


@dataclass
class WebhookStats:
    """Statistics for a webhook."""
    webhook_id: str
    total_deliveries: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0
    pending_deliveries: int = 0
    avg_response_time_ms: float = 0.0
    success_rate: float = 0.0
    last_24h_deliveries: int = 0
    last_24h_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
