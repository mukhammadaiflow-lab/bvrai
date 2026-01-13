"""
Queue & Routing Base Types Module

This module defines core types for call queue management, skill-based routing,
agent availability tracking, and overflow handling.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)


# =============================================================================
# Enums
# =============================================================================


class QueueStatus(str, Enum):
    """Status of a queue."""

    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"
    OVERFLOW = "overflow"


class QueuePriority(str, Enum):
    """Priority levels for queue entries."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class RoutingStrategy(str, Enum):
    """Call routing strategies."""

    ROUND_ROBIN = "round_robin"  # Rotate through available agents
    LEAST_BUSY = "least_busy"  # Route to agent with fewest calls
    SKILLS_BASED = "skills_based"  # Match required skills
    PRIORITY_BASED = "priority_based"  # VIP customers first
    LONGEST_IDLE = "longest_idle"  # Route to agent waiting longest
    WEIGHTED = "weighted"  # Weight-based distribution
    STICKY = "sticky"  # Route to same agent as previous interaction


class AgentStatus(str, Enum):
    """Agent availability status."""

    AVAILABLE = "available"
    BUSY = "busy"
    AWAY = "away"
    OFFLINE = "offline"
    WRAP_UP = "wrap_up"  # Post-call work
    BREAK = "break"
    TRAINING = "training"


class OverflowAction(str, Enum):
    """Actions when queue overflows."""

    VOICEMAIL = "voicemail"
    CALLBACK = "callback"
    TRANSFER = "transfer"  # Transfer to another queue
    MESSAGE = "message"  # Play message and disconnect
    HOLD = "hold"  # Continue holding
    AI_AGENT = "ai_agent"  # Route to AI agent


class CallOutcome(str, Enum):
    """Outcome of a queued call."""

    ANSWERED = "answered"
    ABANDONED = "abandoned"
    VOICEMAIL = "voicemail"
    CALLBACK_SCHEDULED = "callback_scheduled"
    TRANSFERRED = "transferred"
    TIMEOUT = "timeout"
    OVERFLOW = "overflow"


class SkillLevel(str, Enum):
    """Proficiency level for skills."""

    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


# =============================================================================
# Skill Types
# =============================================================================


@dataclass
class Skill:
    """A skill that can be assigned to agents or required for queues."""

    id: str
    name: str
    category: str = "general"
    description: str = ""
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"skill_{uuid.uuid4().hex[:16]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            name=data["name"],
            category=data.get("category", "general"),
            description=data.get("description", ""),
            is_active=data.get("is_active", True),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
        )


@dataclass
class AgentSkill:
    """A skill assigned to an agent with proficiency level."""

    skill_id: str
    skill_name: str
    level: SkillLevel = SkillLevel.INTERMEDIATE
    certified: bool = False
    certified_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    @property
    def level_score(self) -> int:
        """Get numeric score for skill level."""
        scores = {
            SkillLevel.NOVICE: 1,
            SkillLevel.INTERMEDIATE: 2,
            SkillLevel.ADVANCED: 3,
            SkillLevel.EXPERT: 4,
        }
        return scores.get(self.level, 2)

    def is_valid(self) -> bool:
        """Check if skill certification is still valid."""
        if not self.certified:
            return True  # Non-certified skills are always valid
        if self.expires_at and self.expires_at < datetime.utcnow():
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "skill_id": self.skill_id,
            "skill_name": self.skill_name,
            "level": self.level.value,
            "level_score": self.level_score,
            "certified": self.certified,
            "certified_at": self.certified_at.isoformat() if self.certified_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass
class SkillRequirement:
    """A skill requirement for a queue or call."""

    skill_id: str
    skill_name: str
    minimum_level: SkillLevel = SkillLevel.INTERMEDIATE
    required: bool = True  # False = preferred but not required
    weight: float = 1.0  # Importance weight for scoring

    def is_met_by(self, agent_skill: AgentSkill) -> bool:
        """Check if agent skill meets requirement."""
        if agent_skill.skill_id != self.skill_id:
            return False
        if not agent_skill.is_valid():
            return False
        return agent_skill.level_score >= self._minimum_score

    @property
    def _minimum_score(self) -> int:
        """Get minimum required score."""
        scores = {
            SkillLevel.NOVICE: 1,
            SkillLevel.INTERMEDIATE: 2,
            SkillLevel.ADVANCED: 3,
            SkillLevel.EXPERT: 4,
        }
        return scores.get(self.minimum_level, 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "skill_id": self.skill_id,
            "skill_name": self.skill_name,
            "minimum_level": self.minimum_level.value,
            "required": self.required,
            "weight": self.weight,
        }


# =============================================================================
# Agent Types
# =============================================================================


@dataclass
class QueueAgent:
    """An agent that can handle queued calls."""

    id: str
    organization_id: str
    name: str
    email: str

    # Status
    status: AgentStatus = AgentStatus.OFFLINE
    status_reason: str = ""
    status_changed_at: datetime = field(default_factory=datetime.utcnow)

    # Skills
    skills: List[AgentSkill] = field(default_factory=list)

    # Queue assignments
    queue_ids: List[str] = field(default_factory=list)

    # Capacity
    max_concurrent_calls: int = 1
    current_calls: int = 0

    # Priority/Weight
    priority: int = 0  # Higher = preferred
    weight: float = 1.0  # For weighted distribution

    # Performance tracking
    total_calls_handled: int = 0
    avg_handle_time_seconds: float = 0.0
    total_handle_time_seconds: float = 0.0
    last_call_at: Optional[datetime] = None
    last_available_at: Optional[datetime] = None

    # Breaks and scheduling
    on_break_until: Optional[datetime] = None
    shift_start: Optional[datetime] = None
    shift_end: Optional[datetime] = None

    # Metadata
    is_ai_agent: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"qagent_{uuid.uuid4().hex[:20]}"

    @property
    def is_available(self) -> bool:
        """Check if agent is available to take calls."""
        if self.status != AgentStatus.AVAILABLE:
            return False
        if self.current_calls >= self.max_concurrent_calls:
            return False
        if self.on_break_until and self.on_break_until > datetime.utcnow():
            return False
        return True

    @property
    def idle_time(self) -> timedelta:
        """Get how long agent has been idle."""
        if not self.last_call_at:
            return timedelta(0)
        return datetime.utcnow() - self.last_call_at

    @property
    def available_capacity(self) -> int:
        """Get remaining call capacity."""
        return max(0, self.max_concurrent_calls - self.current_calls)

    def has_skill(self, skill_id: str, minimum_level: SkillLevel = SkillLevel.NOVICE) -> bool:
        """Check if agent has a specific skill at minimum level."""
        for skill in self.skills:
            if skill.skill_id == skill_id and skill.is_valid():
                min_score = {
                    SkillLevel.NOVICE: 1,
                    SkillLevel.INTERMEDIATE: 2,
                    SkillLevel.ADVANCED: 3,
                    SkillLevel.EXPERT: 4,
                }.get(minimum_level, 1)
                if skill.level_score >= min_score:
                    return True
        return False

    def get_skill_level(self, skill_id: str) -> Optional[SkillLevel]:
        """Get agent's level for a specific skill."""
        for skill in self.skills:
            if skill.skill_id == skill_id and skill.is_valid():
                return skill.level
        return None

    def meets_requirements(
        self,
        requirements: List[SkillRequirement],
        require_all: bool = True,
    ) -> Tuple[bool, float]:
        """
        Check if agent meets skill requirements.

        Returns:
            Tuple of (meets_requirements, match_score)
        """
        if not requirements:
            return True, 1.0

        total_weight = 0.0
        matched_weight = 0.0
        required_met = True

        for req in requirements:
            total_weight += req.weight

            # Find matching skill
            matching_skill = None
            for agent_skill in self.skills:
                if req.is_met_by(agent_skill):
                    matching_skill = agent_skill
                    break

            if matching_skill:
                # Calculate score based on how much skill exceeds requirement
                excess = matching_skill.level_score - req._minimum_score
                skill_score = 1.0 + (excess * 0.25)  # Bonus for exceeding
                matched_weight += req.weight * skill_score
            elif req.required:
                required_met = False

        match_score = matched_weight / total_weight if total_weight > 0 else 1.0

        if require_all and not required_met:
            return False, match_score

        return True, match_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "name": self.name,
            "email": self.email,
            "status": self.status.value,
            "status_reason": self.status_reason,
            "status_changed_at": self.status_changed_at.isoformat(),
            "skills": [s.to_dict() for s in self.skills],
            "queue_ids": self.queue_ids,
            "max_concurrent_calls": self.max_concurrent_calls,
            "current_calls": self.current_calls,
            "is_available": self.is_available,
            "available_capacity": self.available_capacity,
            "priority": self.priority,
            "weight": self.weight,
            "total_calls_handled": self.total_calls_handled,
            "avg_handle_time_seconds": self.avg_handle_time_seconds,
            "last_call_at": self.last_call_at.isoformat() if self.last_call_at else None,
            "is_ai_agent": self.is_ai_agent,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# =============================================================================
# Queue Types
# =============================================================================


@dataclass
class QueueConfig:
    """Configuration for a call queue."""

    # Routing
    routing_strategy: RoutingStrategy = RoutingStrategy.SKILLS_BASED
    fallback_strategy: RoutingStrategy = RoutingStrategy.ROUND_ROBIN

    # Wait times
    max_wait_time_seconds: int = 300  # 5 minutes
    target_answer_time_seconds: int = 30  # SLA target

    # Queue limits
    max_queue_size: int = 100
    max_concurrent_in_queue: Optional[int] = None

    # Overflow
    overflow_action: OverflowAction = OverflowAction.VOICEMAIL
    overflow_target_queue_id: Optional[str] = None
    overflow_message: str = "All agents are busy. Please leave a message."

    # Callbacks
    callback_enabled: bool = True
    callback_max_attempts: int = 3
    callback_retry_delay_minutes: int = 15

    # Messages
    welcome_message: str = "Thank you for calling. Please hold for the next available agent."
    hold_music_url: Optional[str] = None
    position_announcement_interval_seconds: int = 60
    estimated_wait_announcement: bool = True

    # Business hours
    business_hours_only: bool = False
    business_hours_start: str = "09:00"
    business_hours_end: str = "17:00"
    timezone: str = "UTC"

    # Priority
    vip_priority_boost: int = 2  # Priority boost for VIP callers
    allow_priority_override: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "routing_strategy": self.routing_strategy.value,
            "fallback_strategy": self.fallback_strategy.value,
            "max_wait_time_seconds": self.max_wait_time_seconds,
            "target_answer_time_seconds": self.target_answer_time_seconds,
            "max_queue_size": self.max_queue_size,
            "max_concurrent_in_queue": self.max_concurrent_in_queue,
            "overflow_action": self.overflow_action.value,
            "overflow_target_queue_id": self.overflow_target_queue_id,
            "overflow_message": self.overflow_message,
            "callback_enabled": self.callback_enabled,
            "callback_max_attempts": self.callback_max_attempts,
            "callback_retry_delay_minutes": self.callback_retry_delay_minutes,
            "welcome_message": self.welcome_message,
            "hold_music_url": self.hold_music_url,
            "position_announcement_interval_seconds": self.position_announcement_interval_seconds,
            "estimated_wait_announcement": self.estimated_wait_announcement,
            "business_hours_only": self.business_hours_only,
            "business_hours_start": self.business_hours_start,
            "business_hours_end": self.business_hours_end,
            "timezone": self.timezone,
            "vip_priority_boost": self.vip_priority_boost,
            "allow_priority_override": self.allow_priority_override,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueueConfig":
        """Create from dictionary."""
        return cls(
            routing_strategy=RoutingStrategy(data.get("routing_strategy", "skills_based")),
            fallback_strategy=RoutingStrategy(data.get("fallback_strategy", "round_robin")),
            max_wait_time_seconds=data.get("max_wait_time_seconds", 300),
            target_answer_time_seconds=data.get("target_answer_time_seconds", 30),
            max_queue_size=data.get("max_queue_size", 100),
            max_concurrent_in_queue=data.get("max_concurrent_in_queue"),
            overflow_action=OverflowAction(data.get("overflow_action", "voicemail")),
            overflow_target_queue_id=data.get("overflow_target_queue_id"),
            overflow_message=data.get("overflow_message", ""),
            callback_enabled=data.get("callback_enabled", True),
            callback_max_attempts=data.get("callback_max_attempts", 3),
            callback_retry_delay_minutes=data.get("callback_retry_delay_minutes", 15),
            welcome_message=data.get("welcome_message", ""),
            hold_music_url=data.get("hold_music_url"),
            position_announcement_interval_seconds=data.get("position_announcement_interval_seconds", 60),
            estimated_wait_announcement=data.get("estimated_wait_announcement", True),
            business_hours_only=data.get("business_hours_only", False),
            business_hours_start=data.get("business_hours_start", "09:00"),
            business_hours_end=data.get("business_hours_end", "17:00"),
            timezone=data.get("timezone", "UTC"),
            vip_priority_boost=data.get("vip_priority_boost", 2),
            allow_priority_override=data.get("allow_priority_override", True),
        )


@dataclass
class CallQueue:
    """A call queue for routing incoming calls."""

    id: str
    organization_id: str
    name: str

    # Configuration
    config: QueueConfig = field(default_factory=QueueConfig)

    # Skill requirements
    skill_requirements: List[SkillRequirement] = field(default_factory=list)

    # Status
    status: QueueStatus = QueueStatus.ACTIVE
    status_reason: str = ""

    # Statistics (real-time)
    current_size: int = 0
    total_calls_today: int = 0
    calls_answered_today: int = 0
    calls_abandoned_today: int = 0
    avg_wait_time_today_seconds: float = 0.0
    longest_wait_current_seconds: float = 0.0

    # SLA tracking
    sla_target_percent: float = 80.0  # Target % answered within target time
    sla_current_percent: float = 100.0

    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"queue_{uuid.uuid4().hex[:20]}"

    @property
    def is_accepting_calls(self) -> bool:
        """Check if queue is accepting new calls."""
        if self.status != QueueStatus.ACTIVE:
            return False
        if self.current_size >= self.config.max_queue_size:
            return False
        return True

    @property
    def is_at_capacity(self) -> bool:
        """Check if queue is at maximum capacity."""
        return self.current_size >= self.config.max_queue_size

    @property
    def abandonment_rate(self) -> float:
        """Calculate abandonment rate for today."""
        total = self.calls_answered_today + self.calls_abandoned_today
        if total == 0:
            return 0.0
        return (self.calls_abandoned_today / total) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "name": self.name,
            "config": self.config.to_dict(),
            "skill_requirements": [r.to_dict() for r in self.skill_requirements],
            "status": self.status.value,
            "status_reason": self.status_reason,
            "current_size": self.current_size,
            "total_calls_today": self.total_calls_today,
            "calls_answered_today": self.calls_answered_today,
            "calls_abandoned_today": self.calls_abandoned_today,
            "avg_wait_time_today_seconds": self.avg_wait_time_today_seconds,
            "longest_wait_current_seconds": self.longest_wait_current_seconds,
            "sla_target_percent": self.sla_target_percent,
            "sla_current_percent": self.sla_current_percent,
            "is_accepting_calls": self.is_accepting_calls,
            "is_at_capacity": self.is_at_capacity,
            "abandonment_rate": self.abandonment_rate,
            "description": self.description,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# =============================================================================
# Queue Entry Types
# =============================================================================


@dataclass
class QueueEntry:
    """An entry in a call queue (a waiting call)."""

    id: str
    queue_id: str
    organization_id: str

    # Call info
    call_id: str
    caller_phone: str
    caller_name: Optional[str] = None
    caller_id: Optional[str] = None  # If known customer

    # Priority and routing
    priority: QueuePriority = QueuePriority.NORMAL
    skill_requirements: List[SkillRequirement] = field(default_factory=list)
    preferred_agent_id: Optional[str] = None  # For sticky routing

    # VIP handling
    is_vip: bool = False
    vip_tier: Optional[str] = None

    # Position and timing
    position: int = 0
    entered_at: datetime = field(default_factory=datetime.utcnow)
    estimated_wait_seconds: Optional[float] = None
    max_wait_until: Optional[datetime] = None

    # Status
    outcome: Optional[CallOutcome] = None
    outcome_at: Optional[datetime] = None
    assigned_agent_id: Optional[str] = None
    assigned_at: Optional[datetime] = None

    # Callback handling
    callback_requested: bool = False
    callback_phone: Optional[str] = None
    callback_scheduled_at: Optional[datetime] = None
    callback_attempts: int = 0

    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    source: str = "inbound"  # inbound, callback, transfer
    transfer_from_queue_id: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = f"qentry_{uuid.uuid4().hex[:18]}"

    @property
    def wait_time(self) -> timedelta:
        """Get current wait time."""
        return datetime.utcnow() - self.entered_at

    @property
    def wait_seconds(self) -> float:
        """Get wait time in seconds."""
        return self.wait_time.total_seconds()

    @property
    def is_waiting(self) -> bool:
        """Check if entry is still waiting."""
        return self.outcome is None

    @property
    def has_exceeded_max_wait(self) -> bool:
        """Check if entry has exceeded maximum wait time."""
        if not self.max_wait_until:
            return False
        return datetime.utcnow() > self.max_wait_until

    @property
    def priority_score(self) -> int:
        """Get numeric priority score (higher = more urgent)."""
        base_scores = {
            QueuePriority.CRITICAL: 100,
            QueuePriority.HIGH: 75,
            QueuePriority.NORMAL: 50,
            QueuePriority.LOW: 25,
        }
        score = base_scores.get(self.priority, 50)

        # VIP boost
        if self.is_vip:
            score += 25

        # Wait time boost (up to 25 points for long waits)
        wait_minutes = self.wait_seconds / 60
        score += min(25, wait_minutes * 2.5)

        return int(score)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "queue_id": self.queue_id,
            "organization_id": self.organization_id,
            "call_id": self.call_id,
            "caller_phone": self.caller_phone,
            "caller_name": self.caller_name,
            "caller_id": self.caller_id,
            "priority": self.priority.value,
            "priority_score": self.priority_score,
            "skill_requirements": [r.to_dict() for r in self.skill_requirements],
            "preferred_agent_id": self.preferred_agent_id,
            "is_vip": self.is_vip,
            "vip_tier": self.vip_tier,
            "position": self.position,
            "entered_at": self.entered_at.isoformat(),
            "wait_seconds": self.wait_seconds,
            "estimated_wait_seconds": self.estimated_wait_seconds,
            "outcome": self.outcome.value if self.outcome else None,
            "outcome_at": self.outcome_at.isoformat() if self.outcome_at else None,
            "assigned_agent_id": self.assigned_agent_id,
            "callback_requested": self.callback_requested,
            "callback_scheduled_at": self.callback_scheduled_at.isoformat() if self.callback_scheduled_at else None,
            "context": self.context,
            "source": self.source,
        }


# =============================================================================
# Routing Types
# =============================================================================


@dataclass
class RoutingDecision:
    """Result of a routing decision."""

    entry_id: str
    queue_id: str
    agent_id: Optional[str] = None

    # Decision
    routed: bool = False
    route_type: str = "agent"  # agent, overflow, voicemail, callback

    # Reasoning
    strategy_used: RoutingStrategy = RoutingStrategy.ROUND_ROBIN
    match_score: float = 0.0
    decision_reason: str = ""

    # Overflow info
    overflow_action: Optional[OverflowAction] = None
    overflow_target: Optional[str] = None

    # Timing
    decided_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_id": self.entry_id,
            "queue_id": self.queue_id,
            "agent_id": self.agent_id,
            "routed": self.routed,
            "route_type": self.route_type,
            "strategy_used": self.strategy_used.value,
            "match_score": self.match_score,
            "decision_reason": self.decision_reason,
            "overflow_action": self.overflow_action.value if self.overflow_action else None,
            "overflow_target": self.overflow_target,
            "decided_at": self.decided_at.isoformat(),
        }


@dataclass
class CallbackRequest:
    """A scheduled callback request."""

    id: str
    organization_id: str
    queue_id: str
    original_entry_id: str

    # Contact info
    phone_number: str
    caller_name: Optional[str] = None
    caller_id: Optional[str] = None

    # Scheduling
    scheduled_at: datetime = field(default_factory=datetime.utcnow)
    preferred_time: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # Attempts
    attempts: int = 0
    max_attempts: int = 3
    last_attempt_at: Optional[datetime] = None
    last_attempt_result: Optional[str] = None

    # Status
    completed: bool = False
    completed_at: Optional[datetime] = None
    completed_by_agent_id: Optional[str] = None

    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = f"callback_{uuid.uuid4().hex[:18]}"

    @property
    def can_attempt(self) -> bool:
        """Check if callback can be attempted."""
        if self.completed:
            return False
        if self.attempts >= self.max_attempts:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "queue_id": self.queue_id,
            "original_entry_id": self.original_entry_id,
            "phone_number": self.phone_number,
            "caller_name": self.caller_name,
            "scheduled_at": self.scheduled_at.isoformat(),
            "preferred_time": self.preferred_time.isoformat() if self.preferred_time else None,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "can_attempt": self.can_attempt,
            "completed": self.completed,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "context": self.context,
            "notes": self.notes,
        }


# =============================================================================
# Analytics Types
# =============================================================================


@dataclass
class QueueMetrics:
    """Real-time metrics for a queue."""

    queue_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Current state
    current_waiting: int = 0
    current_talking: int = 0
    available_agents: int = 0
    busy_agents: int = 0

    # Wait times
    avg_wait_time_seconds: float = 0.0
    max_wait_time_seconds: float = 0.0
    min_wait_time_seconds: float = 0.0

    # Handle times
    avg_handle_time_seconds: float = 0.0
    avg_talk_time_seconds: float = 0.0
    avg_wrap_time_seconds: float = 0.0

    # Call volumes
    calls_offered: int = 0
    calls_answered: int = 0
    calls_abandoned: int = 0
    calls_overflowed: int = 0

    # Service level
    service_level_percent: float = 100.0
    service_level_target_seconds: int = 30

    # Abandonment
    abandonment_rate_percent: float = 0.0
    avg_abandon_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "queue_id": self.queue_id,
            "timestamp": self.timestamp.isoformat(),
            "current_waiting": self.current_waiting,
            "current_talking": self.current_talking,
            "available_agents": self.available_agents,
            "busy_agents": self.busy_agents,
            "avg_wait_time_seconds": self.avg_wait_time_seconds,
            "max_wait_time_seconds": self.max_wait_time_seconds,
            "avg_handle_time_seconds": self.avg_handle_time_seconds,
            "calls_offered": self.calls_offered,
            "calls_answered": self.calls_answered,
            "calls_abandoned": self.calls_abandoned,
            "service_level_percent": self.service_level_percent,
            "abandonment_rate_percent": self.abandonment_rate_percent,
        }


@dataclass
class AgentMetrics:
    """Performance metrics for an agent."""

    agent_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_end: Optional[datetime] = None

    # Status time
    available_time_seconds: float = 0.0
    busy_time_seconds: float = 0.0
    away_time_seconds: float = 0.0
    wrap_time_seconds: float = 0.0

    # Call handling
    calls_handled: int = 0
    calls_transferred: int = 0
    calls_missed: int = 0

    # Performance
    avg_handle_time_seconds: float = 0.0
    avg_talk_time_seconds: float = 0.0
    avg_hold_time_seconds: float = 0.0
    avg_wrap_time_seconds: float = 0.0

    # Quality
    avg_customer_rating: float = 0.0
    ratings_count: int = 0
    first_call_resolution_percent: float = 0.0

    # Utilization
    occupancy_percent: float = 0.0  # Time on calls / available time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "period_start": self.period_start.isoformat(),
            "calls_handled": self.calls_handled,
            "calls_transferred": self.calls_transferred,
            "calls_missed": self.calls_missed,
            "avg_handle_time_seconds": self.avg_handle_time_seconds,
            "avg_talk_time_seconds": self.avg_talk_time_seconds,
            "avg_customer_rating": self.avg_customer_rating,
            "occupancy_percent": self.occupancy_percent,
            "available_time_seconds": self.available_time_seconds,
            "busy_time_seconds": self.busy_time_seconds,
        }


# =============================================================================
# Exceptions
# =============================================================================


class QueueRoutingError(Exception):
    """Base exception for queue routing errors."""
    pass


class QueueFullError(QueueRoutingError):
    """Queue is at maximum capacity."""
    pass


class QueueNotFoundError(QueueRoutingError):
    """Queue not found."""
    pass


class AgentNotFoundError(QueueRoutingError):
    """Agent not found."""
    pass


class NoAgentAvailableError(QueueRoutingError):
    """No agents available to handle call."""
    pass


class SkillMismatchError(QueueRoutingError):
    """No agents with required skills available."""
    pass


class QueueClosedError(QueueRoutingError):
    """Queue is closed and not accepting calls."""
    pass


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "QueueStatus",
    "QueuePriority",
    "RoutingStrategy",
    "AgentStatus",
    "OverflowAction",
    "CallOutcome",
    "SkillLevel",
    # Skill types
    "Skill",
    "AgentSkill",
    "SkillRequirement",
    # Agent types
    "QueueAgent",
    # Queue types
    "QueueConfig",
    "CallQueue",
    "QueueEntry",
    # Routing types
    "RoutingDecision",
    "CallbackRequest",
    # Analytics types
    "QueueMetrics",
    "AgentMetrics",
    # Exceptions
    "QueueRoutingError",
    "QueueFullError",
    "QueueNotFoundError",
    "AgentNotFoundError",
    "NoAgentAvailableError",
    "SkillMismatchError",
    "QueueClosedError",
]
