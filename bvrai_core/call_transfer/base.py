"""
Call Transfer & Handoff Base Types

This module defines the core types and data structures for intelligent
call transfer and handoff management, supporting warm transfers, cold
transfers, agent squads, and context-aware routing.

Key Features:
- Multiple transfer types (warm, cold, blind, consultative)
- Transfer targets (human agents, departments, external numbers, AI agents)
- Context passing during transfers
- Skill-based routing
- Queue management integration
- Transfer analytics and monitoring
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)


# =============================================================================
# Transfer Type Enums
# =============================================================================


class TransferType(str, Enum):
    """Types of call transfers."""

    # Warm transfer - Agent stays on line until handoff complete
    WARM = "warm"

    # Cold transfer - Immediate transfer, agent disconnects
    COLD = "cold"

    # Blind transfer - Transfer without announcement
    BLIND = "blind"

    # Consultative transfer - Agent consults with target before transfer
    CONSULTATIVE = "consultative"

    # Conference - All parties stay on the call
    CONFERENCE = "conference"

    # Escalation - Transfer to supervisor/manager
    ESCALATION = "escalation"

    # Department transfer - Transfer to a department queue
    DEPARTMENT = "department"


class TransferTargetType(str, Enum):
    """Types of transfer targets."""

    # Human agents
    HUMAN_AGENT = "human_agent"
    AGENT_GROUP = "agent_group"
    DEPARTMENT = "department"

    # External
    EXTERNAL_NUMBER = "external_number"
    SIP_ENDPOINT = "sip_endpoint"

    # AI agents
    AI_AGENT = "ai_agent"
    AI_SQUAD = "ai_squad"

    # Special
    VOICEMAIL = "voicemail"
    IVR = "ivr"
    CALLBACK_QUEUE = "callback_queue"


class TransferStatus(str, Enum):
    """Status of a transfer."""

    PENDING = "pending"
    INITIATING = "initiating"
    RINGING = "ringing"
    CONNECTED = "connected"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    NO_ANSWER = "no_answer"
    BUSY = "busy"


class TransferReason(str, Enum):
    """Reasons for transfer."""

    # Customer-initiated
    CUSTOMER_REQUEST = "customer_request"
    LANGUAGE_PREFERENCE = "language_preference"

    # Skill-based
    TECHNICAL_EXPERTISE = "technical_expertise"
    BILLING_ISSUE = "billing_issue"
    SALES_OPPORTUNITY = "sales_opportunity"
    COMPLEX_ISSUE = "complex_issue"

    # Escalation
    CUSTOMER_FRUSTRATED = "customer_frustrated"
    SUPERVISOR_NEEDED = "supervisor_needed"
    POLICY_EXCEPTION = "policy_exception"

    # Operational
    AGENT_UNAVAILABLE = "agent_unavailable"
    QUEUE_OVERFLOW = "queue_overflow"
    AFTER_HOURS = "after_hours"

    # AI-specific
    AI_LIMITATION = "ai_limitation"
    HUMAN_REQUIRED = "human_required"

    # Other
    OTHER = "other"


class TransferPriority(str, Enum):
    """Priority of transfer."""

    CRITICAL = "critical"  # VIP customer, urgent issue
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


# =============================================================================
# Transfer Target Types
# =============================================================================


@dataclass
class HumanAgent:
    """A human agent who can receive transfers."""

    id: str
    name: str
    email: str

    # Status
    is_available: bool = True
    is_online: bool = True
    current_status: str = "available"  # available, busy, away, offline

    # Skills
    skills: List[str] = field(default_factory=list)
    skill_levels: Dict[str, int] = field(default_factory=dict)  # skill -> 1-5

    # Languages
    languages: List[str] = field(default_factory=list)

    # Capacity
    max_concurrent_calls: int = 1
    current_calls: int = 0

    # Performance
    average_handle_time_seconds: float = 0.0
    customer_satisfaction_score: float = 0.0

    # Contact
    extension: str = ""
    direct_number: str = ""
    sip_uri: str = ""

    # Groups
    department: str = ""
    team: str = ""
    agent_groups: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = f"agent_{uuid.uuid4().hex[:12]}"

    @property
    def can_accept_call(self) -> bool:
        """Check if agent can accept a new call."""
        return (
            self.is_available
            and self.is_online
            and self.current_calls < self.max_concurrent_calls
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "is_available": self.is_available,
            "is_online": self.is_online,
            "current_status": self.current_status,
            "skills": self.skills,
            "languages": self.languages,
            "can_accept_call": self.can_accept_call,
            "department": self.department,
            "team": self.team,
        }


@dataclass
class AgentGroup:
    """A group of agents that can receive transfers."""

    id: str
    name: str
    description: str = ""

    # Members
    agent_ids: List[str] = field(default_factory=list)

    # Skills (inherited by all members)
    skills: List[str] = field(default_factory=list)

    # Routing
    routing_strategy: str = "round_robin"  # round_robin, least_busy, skill_based

    # Queue settings
    queue_enabled: bool = True
    max_queue_size: int = 50
    max_wait_time_seconds: int = 300

    # Hours of operation
    hours_of_operation: Dict[str, Any] = field(default_factory=dict)

    # Fallback
    fallback_group_id: Optional[str] = None
    overflow_group_id: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = f"group_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "agent_count": len(self.agent_ids),
            "skills": self.skills,
            "routing_strategy": self.routing_strategy,
            "queue_enabled": self.queue_enabled,
        }


@dataclass
class Department:
    """A department that can receive transfers."""

    id: str
    name: str
    description: str = ""

    # Contact
    main_number: str = ""
    extension: str = ""
    sip_uri: str = ""

    # Agent groups
    agent_group_ids: List[str] = field(default_factory=list)

    # Skills handled
    skills: List[str] = field(default_factory=list)

    # Queue settings
    queue_id: Optional[str] = None

    # Hours
    is_24_7: bool = False
    hours_of_operation: Dict[str, Any] = field(default_factory=dict)
    timezone: str = "UTC"

    # Fallback
    after_hours_destination: Optional[str] = None
    overflow_destination: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = f"dept_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "main_number": self.main_number,
            "skills": self.skills,
            "is_24_7": self.is_24_7,
        }


@dataclass
class ExternalDestination:
    """An external phone number or SIP endpoint."""

    id: str
    name: str

    # Destination
    phone_number: str = ""
    sip_uri: str = ""

    # Type
    destination_type: str = "phone"  # phone, sip, pstn

    # Settings
    caller_id: str = ""  # Caller ID to display
    timeout_seconds: int = 30

    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = f"ext_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "phone_number": self.phone_number,
            "sip_uri": self.sip_uri,
            "destination_type": self.destination_type,
        }


@dataclass
class AIAgentTarget:
    """An AI agent that can receive transfers."""

    id: str
    name: str
    agent_id: str  # Reference to the AI agent

    # Capabilities
    skills: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)

    # Capacity
    max_concurrent_calls: int = 100
    current_calls: int = 0

    # Status
    is_available: bool = True
    is_healthy: bool = True

    def __post_init__(self):
        if not self.id:
            self.id = f"ai_target_{uuid.uuid4().hex[:12]}"

    @property
    def can_accept_call(self) -> bool:
        """Check if AI agent can accept a new call."""
        return (
            self.is_available
            and self.is_healthy
            and self.current_calls < self.max_concurrent_calls
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "agent_id": self.agent_id,
            "skills": self.skills,
            "languages": self.languages,
            "can_accept_call": self.can_accept_call,
        }


# =============================================================================
# Transfer Target Union
# =============================================================================


@dataclass
class TransferTarget:
    """Generic transfer target."""

    id: str
    target_type: TransferTargetType
    name: str

    # Target reference
    target_id: str = ""  # ID of the specific target

    # Direct contact (for external/sip)
    phone_number: str = ""
    sip_uri: str = ""
    extension: str = ""

    # Context for the target
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = f"target_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "target_type": self.target_type.value,
            "name": self.name,
            "target_id": self.target_id,
            "phone_number": self.phone_number,
            "sip_uri": self.sip_uri,
        }


# =============================================================================
# Transfer Context Types
# =============================================================================


@dataclass
class ConversationSummaryContext:
    """Summary of conversation to pass during transfer."""

    # Brief summary
    summary: str = ""

    # Key information
    customer_name: str = ""
    customer_id: str = ""
    customer_phone: str = ""
    customer_email: str = ""

    # Issue details
    issue_type: str = ""
    issue_description: str = ""
    issue_severity: str = ""

    # Sentiment
    customer_sentiment: str = ""  # positive, neutral, negative
    sentiment_score: float = 0.0

    # Key entities mentioned
    entities: Dict[str, Any] = field(default_factory=dict)

    # Action items
    pending_actions: List[str] = field(default_factory=list)
    completed_actions: List[str] = field(default_factory=list)

    # Notes
    agent_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": self.summary,
            "customer_name": self.customer_name,
            "customer_id": self.customer_id,
            "issue_type": self.issue_type,
            "issue_description": self.issue_description,
            "customer_sentiment": self.customer_sentiment,
            "pending_actions": self.pending_actions,
            "agent_notes": self.agent_notes,
        }


@dataclass
class TransferContext:
    """Complete context for a transfer."""

    # Conversation context
    conversation_summary: ConversationSummaryContext = field(
        default_factory=ConversationSummaryContext
    )

    # Call metadata
    call_id: str = ""
    call_duration_seconds: float = 0.0
    call_start_time: Optional[datetime] = None

    # Previous agents
    previous_agents: List[str] = field(default_factory=list)
    transfer_count: int = 0

    # Queue info
    queue_wait_time_seconds: float = 0.0
    queue_position: int = 0

    # Priority
    priority: TransferPriority = TransferPriority.NORMAL
    is_vip: bool = False

    # Skills needed
    required_skills: List[str] = field(default_factory=list)
    preferred_language: str = ""

    # Custom data
    custom_data: Dict[str, Any] = field(default_factory=dict)

    # Transcript (optional, for warm transfers)
    include_transcript: bool = False
    transcript_url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conversation_summary": self.conversation_summary.to_dict(),
            "call_id": self.call_id,
            "call_duration_seconds": self.call_duration_seconds,
            "previous_agents": self.previous_agents,
            "transfer_count": self.transfer_count,
            "priority": self.priority.value,
            "is_vip": self.is_vip,
            "required_skills": self.required_skills,
            "preferred_language": self.preferred_language,
            "include_transcript": self.include_transcript,
        }


# =============================================================================
# Transfer Request and Result Types
# =============================================================================


@dataclass
class TransferRequest:
    """Request to transfer a call."""

    id: str
    organization_id: str
    call_id: str

    # Transfer details
    transfer_type: TransferType = TransferType.WARM
    reason: TransferReason = TransferReason.OTHER
    priority: TransferPriority = TransferPriority.NORMAL

    # Source
    source_agent_id: str = ""
    source_agent_type: str = ""  # "ai" or "human"

    # Target
    target: Optional[TransferTarget] = None

    # Alternative targets (for skill-based routing)
    target_skills: List[str] = field(default_factory=list)
    target_department: str = ""
    target_agent_group: str = ""

    # Context
    context: TransferContext = field(default_factory=TransferContext)

    # Options
    announce_transfer: bool = True
    play_hold_music: bool = True
    timeout_seconds: int = 60
    max_retries: int = 2

    # Callback options
    offer_callback: bool = False
    callback_number: str = ""

    # Status
    status: TransferStatus = TransferStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"transfer_{uuid.uuid4().hex[:14]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "call_id": self.call_id,
            "transfer_type": self.transfer_type.value,
            "reason": self.reason.value,
            "priority": self.priority.value,
            "source_agent_id": self.source_agent_id,
            "target": self.target.to_dict() if self.target else None,
            "target_skills": self.target_skills,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class TransferResult:
    """Result of a transfer attempt."""

    request_id: str
    success: bool
    status: TransferStatus

    # Target info
    connected_to: Optional[str] = None  # ID of agent/target connected
    connected_name: str = ""
    target_type: Optional[TransferTargetType] = None

    # Timing
    ring_time_seconds: float = 0.0
    connect_time_seconds: float = 0.0
    total_time_seconds: float = 0.0

    # Error info
    error_code: str = ""
    error_message: str = ""

    # Retry info
    attempt_number: int = 1
    max_attempts: int = 1

    # Metadata
    completed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "status": self.status.value,
            "connected_to": self.connected_to,
            "connected_name": self.connected_name,
            "target_type": self.target_type.value if self.target_type else None,
            "ring_time_seconds": self.ring_time_seconds,
            "connect_time_seconds": self.connect_time_seconds,
            "error_message": self.error_message,
            "attempt_number": self.attempt_number,
            "completed_at": self.completed_at.isoformat(),
        }


# =============================================================================
# Routing Rule Types
# =============================================================================


class RoutingStrategy(str, Enum):
    """Routing strategies for finding transfer targets."""

    ROUND_ROBIN = "round_robin"
    LEAST_BUSY = "least_busy"
    LONGEST_IDLE = "longest_idle"
    SKILL_BASED = "skill_based"
    PRIORITY_BASED = "priority_based"
    WEIGHTED_RANDOM = "weighted_random"
    DIRECT = "direct"  # Direct to specific agent


class RoutingConditionType(str, Enum):
    """Types of routing conditions."""

    SKILL_MATCH = "skill_match"
    LANGUAGE_MATCH = "language_match"
    DEPARTMENT_MATCH = "department_match"
    TIME_OF_DAY = "time_of_day"
    QUEUE_LENGTH = "queue_length"
    WAIT_TIME = "wait_time"
    CUSTOMER_TYPE = "customer_type"
    SENTIMENT = "sentiment"
    CUSTOM = "custom"


@dataclass
class RoutingCondition:
    """A condition for routing rules."""

    condition_type: RoutingConditionType
    operator: str = "equals"  # equals, contains, greater_than, less_than
    value: Any = None
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "condition_type": self.condition_type.value,
            "operator": self.operator,
            "value": self.value,
            "weight": self.weight,
        }


@dataclass
class RoutingRule:
    """A routing rule for transfers."""

    id: str
    name: str
    organization_id: str

    # Priority (lower = higher priority)
    priority: int = 100

    # Conditions (all must match)
    conditions: List[RoutingCondition] = field(default_factory=list)

    # Target selection
    strategy: RoutingStrategy = RoutingStrategy.SKILL_BASED
    target_type: TransferTargetType = TransferTargetType.AGENT_GROUP
    target_ids: List[str] = field(default_factory=list)  # Candidate targets

    # Fallback
    fallback_rule_id: Optional[str] = None
    fallback_target_id: Optional[str] = None

    # Settings
    is_active: bool = True
    max_queue_time_seconds: int = 300

    # Schedule
    active_hours: Optional[Dict[str, Any]] = None
    timezone: str = "UTC"

    # Metadata
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"rule_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "priority": self.priority,
            "conditions": [c.to_dict() for c in self.conditions],
            "strategy": self.strategy.value,
            "target_type": self.target_type.value,
            "target_ids": self.target_ids,
            "is_active": self.is_active,
            "description": self.description,
        }


# =============================================================================
# Transfer Analytics Types
# =============================================================================


@dataclass
class TransferMetrics:
    """Metrics for transfer analytics."""

    # Volume
    total_transfers: int = 0
    successful_transfers: int = 0
    failed_transfers: int = 0

    # By type
    transfers_by_type: Dict[TransferType, int] = field(default_factory=dict)
    transfers_by_reason: Dict[TransferReason, int] = field(default_factory=dict)

    # Timing
    average_ring_time_seconds: float = 0.0
    average_connect_time_seconds: float = 0.0
    average_total_time_seconds: float = 0.0

    # Success rate
    success_rate: float = 0.0
    first_attempt_success_rate: float = 0.0

    # Queue metrics
    average_queue_wait_seconds: float = 0.0
    max_queue_wait_seconds: float = 0.0
    abandoned_in_queue: int = 0

    # Period
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None

    def calculate_success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_transfers == 0:
            return 0.0
        return self.successful_transfers / self.total_transfers

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_transfers": self.total_transfers,
            "successful_transfers": self.successful_transfers,
            "failed_transfers": self.failed_transfers,
            "success_rate": self.success_rate or self.calculate_success_rate(),
            "average_ring_time_seconds": self.average_ring_time_seconds,
            "average_connect_time_seconds": self.average_connect_time_seconds,
            "transfers_by_type": {
                k.value: v for k, v in self.transfers_by_type.items()
            },
        }


@dataclass
class AgentTransferMetrics:
    """Transfer metrics for a specific agent."""

    agent_id: str
    agent_name: str

    # Transfers initiated
    transfers_out: int = 0
    transfer_out_reasons: Dict[TransferReason, int] = field(default_factory=dict)

    # Transfers received
    transfers_in: int = 0
    transfers_accepted: int = 0
    transfers_rejected: int = 0
    transfers_missed: int = 0

    # Timing
    average_accept_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "transfers_out": self.transfers_out,
            "transfers_in": self.transfers_in,
            "transfers_accepted": self.transfers_accepted,
            "transfers_rejected": self.transfers_rejected,
            "transfers_missed": self.transfers_missed,
            "average_accept_time_seconds": self.average_accept_time_seconds,
        }


# =============================================================================
# Exceptions
# =============================================================================


class TransferError(Exception):
    """Base exception for transfer errors."""
    pass


class TransferTargetNotFoundError(TransferError):
    """Transfer target not found."""
    pass


class TransferTargetUnavailableError(TransferError):
    """Transfer target is unavailable."""
    pass


class TransferTimeoutError(TransferError):
    """Transfer timed out."""
    pass


class TransferRejectedError(TransferError):
    """Transfer was rejected."""
    pass


class NoAvailableAgentsError(TransferError):
    """No agents available for transfer."""
    pass


class RoutingRuleNotFoundError(TransferError):
    """Routing rule not found."""
    pass


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "TransferType",
    "TransferTargetType",
    "TransferStatus",
    "TransferReason",
    "TransferPriority",
    "RoutingStrategy",
    "RoutingConditionType",
    # Target types
    "HumanAgent",
    "AgentGroup",
    "Department",
    "ExternalDestination",
    "AIAgentTarget",
    "TransferTarget",
    # Context types
    "ConversationSummaryContext",
    "TransferContext",
    # Request/Result types
    "TransferRequest",
    "TransferResult",
    # Routing types
    "RoutingCondition",
    "RoutingRule",
    # Analytics types
    "TransferMetrics",
    "AgentTransferMetrics",
    # Exceptions
    "TransferError",
    "TransferTargetNotFoundError",
    "TransferTargetUnavailableError",
    "TransferTimeoutError",
    "TransferRejectedError",
    "NoAvailableAgentsError",
    "RoutingRuleNotFoundError",
]
