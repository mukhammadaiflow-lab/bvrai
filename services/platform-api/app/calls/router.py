"""
Call Router

Intelligent call routing with:
- Skill-based routing
- Time-based routing
- Percentage-based routing
- Priority routing
- Load balancing
"""

from typing import Optional, Dict, Any, List, Callable, Awaitable, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging
import random
import re

logger = logging.getLogger(__name__)


class RoutingStrategy(str, Enum):
    """Routing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_BUSY = "least_busy"
    SKILL_BASED = "skill_based"
    PRIORITY = "priority"
    RANDOM = "random"
    WEIGHTED = "weighted"
    TIME_BASED = "time_based"
    PERCENTAGE = "percentage"


@dataclass
class RoutingContext:
    """Context for routing decisions."""
    call_id: str
    caller_number: str
    called_number: str
    tenant_id: str

    # Caller info
    caller_name: Optional[str] = None
    caller_type: Optional[str] = None
    caller_priority: int = 0
    caller_language: str = "en"

    # Call info
    call_type: str = "inbound"
    skill_requirements: List[str] = field(default_factory=list)
    preferred_agents: List[str] = field(default_factory=list)

    # Timing
    timestamp: datetime = field(default_factory=datetime.utcnow)
    wait_time_seconds: float = 0.0
    max_wait_seconds: float = 300.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches_pattern(self, pattern: str, field: str = "caller_number") -> bool:
        """Check if field matches pattern."""
        value = getattr(self, field, "")
        if not value:
            return False

        # Support wildcards
        regex = pattern.replace("*", ".*").replace("?", ".")
        return bool(re.match(regex, value))


@dataclass
class RoutingTarget:
    """Routing target (agent/queue/destination)."""
    target_id: str
    target_type: str  # agent, queue, ivr, external
    priority: int = 0
    weight: float = 1.0

    # Status
    available: bool = True
    current_calls: int = 0
    max_calls: int = 10

    # Skills
    skills: List[str] = field(default_factory=list)
    skill_levels: Dict[str, int] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def load(self) -> float:
        """Get current load (0-1)."""
        if self.max_calls == 0:
            return 1.0
        return self.current_calls / self.max_calls

    @property
    def available_capacity(self) -> int:
        """Get available capacity."""
        return max(0, self.max_calls - self.current_calls)

    def has_skill(self, skill: str, min_level: int = 1) -> bool:
        """Check if target has skill at level."""
        if skill not in self.skills:
            return False
        return self.skill_levels.get(skill, 0) >= min_level


@dataclass
class RoutingResult:
    """Result of routing decision."""
    success: bool
    target: Optional[RoutingTarget] = None
    targets: List[RoutingTarget] = field(default_factory=list)
    strategy: Optional[RoutingStrategy] = None
    reason: str = ""
    fallback_used: bool = False
    processing_time_ms: float = 0.0


@dataclass
class RoutingRule:
    """Routing rule definition."""
    rule_id: str
    name: str
    priority: int = 0
    enabled: bool = True

    # Conditions
    conditions: Dict[str, Any] = field(default_factory=dict)

    # Actions
    target_type: str = "queue"
    target_id: Optional[str] = None
    target_ids: List[str] = field(default_factory=list)
    strategy: RoutingStrategy = RoutingStrategy.ROUND_ROBIN

    # Schedule
    active_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    active_hours_start: time = time(0, 0)
    active_hours_end: time = time(23, 59)

    def is_active(self, dt: Optional[datetime] = None) -> bool:
        """Check if rule is currently active."""
        if not self.enabled:
            return False

        dt = dt or datetime.utcnow()

        # Check day
        if dt.weekday() not in self.active_days:
            return False

        # Check time
        current_time = dt.time()
        if self.active_hours_start <= self.active_hours_end:
            return self.active_hours_start <= current_time <= self.active_hours_end
        else:
            # Spans midnight
            return current_time >= self.active_hours_start or current_time <= self.active_hours_end

    def matches(self, context: RoutingContext) -> bool:
        """Check if rule matches context."""
        if not self.is_active(context.timestamp):
            return False

        for key, value in self.conditions.items():
            if not self._check_condition(context, key, value):
                return False

        return True

    def _check_condition(
        self,
        context: RoutingContext,
        key: str,
        value: Any,
    ) -> bool:
        """Check single condition."""
        if key == "caller_pattern":
            return context.matches_pattern(value, "caller_number")

        if key == "called_pattern":
            return context.matches_pattern(value, "called_number")

        if key == "tenant_id":
            return context.tenant_id == value

        if key == "caller_type":
            return context.caller_type == value

        if key == "min_priority":
            return context.caller_priority >= value

        if key == "language":
            return context.caller_language == value

        if key == "skills":
            required = set(value) if isinstance(value, list) else {value}
            return required.issubset(set(context.skill_requirements))

        if key == "call_type":
            return context.call_type == value

        # Check metadata
        if key.startswith("metadata."):
            meta_key = key[9:]
            return context.metadata.get(meta_key) == value

        # Check context attribute
        if hasattr(context, key):
            return getattr(context, key) == value

        return True


class CallRouter(ABC):
    """Abstract call router."""

    @abstractmethod
    async def route(
        self,
        context: RoutingContext,
        targets: List[RoutingTarget],
    ) -> RoutingResult:
        """Route call to target(s)."""
        pass

    @property
    @abstractmethod
    def strategy(self) -> RoutingStrategy:
        """Get routing strategy."""
        pass


class RoundRobinRouter(CallRouter):
    """Round-robin routing."""

    def __init__(self):
        self._index = 0
        self._lock = asyncio.Lock()

    @property
    def strategy(self) -> RoutingStrategy:
        return RoutingStrategy.ROUND_ROBIN

    async def route(
        self,
        context: RoutingContext,
        targets: List[RoutingTarget],
    ) -> RoutingResult:
        """Route using round-robin."""
        available = [t for t in targets if t.available and t.available_capacity > 0]

        if not available:
            return RoutingResult(success=False, reason="No available targets")

        async with self._lock:
            self._index = self._index % len(available)
            target = available[self._index]
            self._index += 1

        return RoutingResult(
            success=True,
            target=target,
            targets=[target],
            strategy=self.strategy,
        )


class LeastBusyRouter(CallRouter):
    """Least busy (least calls) routing."""

    @property
    def strategy(self) -> RoutingStrategy:
        return RoutingStrategy.LEAST_BUSY

    async def route(
        self,
        context: RoutingContext,
        targets: List[RoutingTarget],
    ) -> RoutingResult:
        """Route to least busy target."""
        available = [t for t in targets if t.available and t.available_capacity > 0]

        if not available:
            return RoutingResult(success=False, reason="No available targets")

        # Sort by load (ascending)
        sorted_targets = sorted(available, key=lambda t: t.load)
        target = sorted_targets[0]

        return RoutingResult(
            success=True,
            target=target,
            targets=[target],
            strategy=self.strategy,
        )


class SkillBasedRouter(CallRouter):
    """Skill-based routing."""

    def __init__(self, min_skill_level: int = 1):
        self.min_skill_level = min_skill_level

    @property
    def strategy(self) -> RoutingStrategy:
        return RoutingStrategy.SKILL_BASED

    async def route(
        self,
        context: RoutingContext,
        targets: List[RoutingTarget],
    ) -> RoutingResult:
        """Route based on skill requirements."""
        required_skills = set(context.skill_requirements)

        # Filter by availability and skills
        matching = []
        for target in targets:
            if not target.available or target.available_capacity <= 0:
                continue

            target_skills = set(target.skills)
            if required_skills.issubset(target_skills):
                # Check skill levels
                meets_levels = all(
                    target.skill_levels.get(skill, 0) >= self.min_skill_level
                    for skill in required_skills
                )
                if meets_levels:
                    matching.append(target)

        if not matching:
            return RoutingResult(
                success=False,
                reason=f"No targets with required skills: {required_skills}",
            )

        # Sort by total skill level and load
        def score(t: RoutingTarget) -> float:
            skill_score = sum(
                t.skill_levels.get(s, 0) for s in required_skills
            )
            load_penalty = t.load * 10
            return skill_score - load_penalty

        sorted_targets = sorted(matching, key=score, reverse=True)
        target = sorted_targets[0]

        return RoutingResult(
            success=True,
            target=target,
            targets=sorted_targets,
            strategy=self.strategy,
        )


class PriorityRouter(CallRouter):
    """Priority-based routing."""

    @property
    def strategy(self) -> RoutingStrategy:
        return RoutingStrategy.PRIORITY

    async def route(
        self,
        context: RoutingContext,
        targets: List[RoutingTarget],
    ) -> RoutingResult:
        """Route based on target priority."""
        available = [t for t in targets if t.available and t.available_capacity > 0]

        if not available:
            return RoutingResult(success=False, reason="No available targets")

        # Sort by priority (descending)
        sorted_targets = sorted(available, key=lambda t: t.priority, reverse=True)
        target = sorted_targets[0]

        return RoutingResult(
            success=True,
            target=target,
            targets=[target],
            strategy=self.strategy,
        )


class WeightedRouter(CallRouter):
    """Weighted random routing."""

    @property
    def strategy(self) -> RoutingStrategy:
        return RoutingStrategy.WEIGHTED

    async def route(
        self,
        context: RoutingContext,
        targets: List[RoutingTarget],
    ) -> RoutingResult:
        """Route using weighted random selection."""
        available = [t for t in targets if t.available and t.available_capacity > 0]

        if not available:
            return RoutingResult(success=False, reason="No available targets")

        # Calculate weights
        total_weight = sum(t.weight for t in available)
        if total_weight == 0:
            # Equal weights
            target = random.choice(available)
        else:
            # Weighted selection
            r = random.uniform(0, total_weight)
            cumulative = 0
            target = available[0]

            for t in available:
                cumulative += t.weight
                if r <= cumulative:
                    target = t
                    break

        return RoutingResult(
            success=True,
            target=target,
            targets=[target],
            strategy=self.strategy,
        )


class TimeBasedRouter(CallRouter):
    """Time-based routing."""

    def __init__(self):
        self._schedules: Dict[str, Dict[str, Any]] = {}

    @property
    def strategy(self) -> RoutingStrategy:
        return RoutingStrategy.TIME_BASED

    def add_schedule(
        self,
        name: str,
        target_id: str,
        days: List[int],
        start_time: time,
        end_time: time,
        priority: int = 0,
    ) -> None:
        """Add schedule entry."""
        self._schedules[name] = {
            "target_id": target_id,
            "days": days,
            "start_time": start_time,
            "end_time": end_time,
            "priority": priority,
        }

    def _get_active_schedule(self, dt: datetime) -> Optional[Dict[str, Any]]:
        """Get currently active schedule."""
        active = []

        for schedule in self._schedules.values():
            if dt.weekday() not in schedule["days"]:
                continue

            current_time = dt.time()
            start = schedule["start_time"]
            end = schedule["end_time"]

            if start <= end:
                is_active = start <= current_time <= end
            else:
                is_active = current_time >= start or current_time <= end

            if is_active:
                active.append(schedule)

        if not active:
            return None

        # Return highest priority
        return max(active, key=lambda s: s["priority"])

    async def route(
        self,
        context: RoutingContext,
        targets: List[RoutingTarget],
    ) -> RoutingResult:
        """Route based on time schedule."""
        schedule = self._get_active_schedule(context.timestamp)

        if not schedule:
            return RoutingResult(success=False, reason="No active schedule")

        target_id = schedule["target_id"]
        target = next((t for t in targets if t.target_id == target_id), None)

        if not target or not target.available:
            return RoutingResult(
                success=False,
                reason=f"Scheduled target {target_id} not available",
            )

        return RoutingResult(
            success=True,
            target=target,
            targets=[target],
            strategy=self.strategy,
        )


class PercentageRouter(CallRouter):
    """Percentage-based routing (A/B testing)."""

    def __init__(self):
        self._percentages: Dict[str, float] = {}
        self._call_counts: Dict[str, int] = {}

    @property
    def strategy(self) -> RoutingStrategy:
        return RoutingStrategy.PERCENTAGE

    def set_percentage(self, target_id: str, percentage: float) -> None:
        """Set routing percentage for target (0-100)."""
        self._percentages[target_id] = max(0, min(100, percentage))
        self._call_counts.setdefault(target_id, 0)

    async def route(
        self,
        context: RoutingContext,
        targets: List[RoutingTarget],
    ) -> RoutingResult:
        """Route based on percentages."""
        available = [
            t for t in targets
            if t.target_id in self._percentages
            and t.available
            and t.available_capacity > 0
        ]

        if not available:
            return RoutingResult(success=False, reason="No available targets")

        # Calculate cumulative percentages
        total = sum(self._percentages.get(t.target_id, 0) for t in available)
        if total == 0:
            target = random.choice(available)
        else:
            r = random.uniform(0, total)
            cumulative = 0
            target = available[0]

            for t in available:
                cumulative += self._percentages.get(t.target_id, 0)
                if r <= cumulative:
                    target = t
                    break

        self._call_counts[target.target_id] = self._call_counts.get(target.target_id, 0) + 1

        return RoutingResult(
            success=True,
            target=target,
            targets=[target],
            strategy=self.strategy,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        total = sum(self._call_counts.values())
        return {
            "percentages": self._percentages.copy(),
            "actual_distribution": {
                tid: count / max(1, total) * 100
                for tid, count in self._call_counts.items()
            },
            "total_calls": total,
        }


class CompositeRouter(CallRouter):
    """Composite router with fallback chain."""

    def __init__(self, routers: Optional[List[CallRouter]] = None):
        self._routers = routers or []

    @property
    def strategy(self) -> RoutingStrategy:
        return RoutingStrategy.ROUND_ROBIN

    def add_router(self, router: CallRouter) -> "CompositeRouter":
        """Add router to chain."""
        self._routers.append(router)
        return self

    async def route(
        self,
        context: RoutingContext,
        targets: List[RoutingTarget],
    ) -> RoutingResult:
        """Try routers in sequence until one succeeds."""
        for i, router in enumerate(self._routers):
            result = await router.route(context, targets)
            if result.success:
                result.fallback_used = i > 0
                return result

        return RoutingResult(
            success=False,
            reason="All routers failed to find target",
        )


class RuleBasedRouter(CallRouter):
    """Rule-based routing with configurable rules."""

    def __init__(self):
        self._rules: List[RoutingRule] = []
        self._routers: Dict[RoutingStrategy, CallRouter] = {
            RoutingStrategy.ROUND_ROBIN: RoundRobinRouter(),
            RoutingStrategy.LEAST_BUSY: LeastBusyRouter(),
            RoutingStrategy.SKILL_BASED: SkillBasedRouter(),
            RoutingStrategy.PRIORITY: PriorityRouter(),
            RoutingStrategy.WEIGHTED: WeightedRouter(),
            RoutingStrategy.RANDOM: WeightedRouter(),  # Random is weighted with equal weights
        }

    @property
    def strategy(self) -> RoutingStrategy:
        return RoutingStrategy.SKILL_BASED

    def add_rule(self, rule: RoutingRule) -> None:
        """Add routing rule."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove rule by ID."""
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.rule_id != rule_id]
        return len(self._rules) < before

    async def route(
        self,
        context: RoutingContext,
        targets: List[RoutingTarget],
    ) -> RoutingResult:
        """Route based on matching rules."""
        import time as time_module
        start = time_module.time()

        # Find matching rule
        matched_rule = None
        for rule in self._rules:
            if rule.matches(context):
                matched_rule = rule
                break

        if not matched_rule:
            return RoutingResult(
                success=False,
                reason="No matching routing rule",
                processing_time_ms=(time_module.time() - start) * 1000,
            )

        # Get targets for rule
        rule_targets = targets
        if matched_rule.target_ids:
            rule_targets = [
                t for t in targets
                if t.target_id in matched_rule.target_ids
            ]
        elif matched_rule.target_id:
            rule_targets = [
                t for t in targets
                if t.target_id == matched_rule.target_id
            ]

        if not rule_targets:
            return RoutingResult(
                success=False,
                reason=f"No targets found for rule {matched_rule.rule_id}",
                processing_time_ms=(time_module.time() - start) * 1000,
            )

        # Use rule's strategy
        router = self._routers.get(matched_rule.strategy)
        if not router:
            router = self._routers[RoutingStrategy.ROUND_ROBIN]

        result = await router.route(context, rule_targets)
        result.processing_time_ms = (time_module.time() - start) * 1000

        return result

    def get_rules(self) -> List[RoutingRule]:
        """Get all rules."""
        return list(self._rules)


class RouterManager:
    """
    Manages call routing.
    """

    def __init__(self):
        self._targets: Dict[str, RoutingTarget] = {}
        self._router = RuleBasedRouter()
        self._fallback_router = LeastBusyRouter()
        self._lock = asyncio.Lock()

        # Statistics
        self._routes_attempted = 0
        self._routes_succeeded = 0

    def register_target(self, target: RoutingTarget) -> None:
        """Register routing target."""
        self._targets[target.target_id] = target

    def unregister_target(self, target_id: str) -> bool:
        """Unregister routing target."""
        return self._targets.pop(target_id, None) is not None

    def update_target_status(
        self,
        target_id: str,
        available: Optional[bool] = None,
        current_calls: Optional[int] = None,
    ) -> bool:
        """Update target status."""
        target = self._targets.get(target_id)
        if not target:
            return False

        if available is not None:
            target.available = available
        if current_calls is not None:
            target.current_calls = current_calls

        return True

    def add_rule(self, rule: RoutingRule) -> None:
        """Add routing rule."""
        self._router.add_rule(rule)

    async def route(self, context: RoutingContext) -> RoutingResult:
        """Route call."""
        self._routes_attempted += 1

        targets = list(self._targets.values())

        # Try primary router
        result = await self._router.route(context, targets)

        # Try fallback if primary fails
        if not result.success:
            result = await self._fallback_router.route(context, targets)
            result.fallback_used = True

        if result.success:
            self._routes_succeeded += 1

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            "targets": len(self._targets),
            "available_targets": sum(
                1 for t in self._targets.values() if t.available
            ),
            "rules": len(self._router.get_rules()),
            "routes_attempted": self._routes_attempted,
            "routes_succeeded": self._routes_succeeded,
            "success_rate": self._routes_succeeded / max(1, self._routes_attempted),
        }
