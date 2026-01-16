"""
Call Routing Module

This module provides call routing capabilities including
rule-based routing, time-based routing, and load balancing.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

from .base import Call, CallDirection


logger = logging.getLogger(__name__)


class RoutingStrategy(str, Enum):
    """Call routing strategies."""
    SEQUENTIAL = "sequential"  # Try destinations in order
    ROUND_ROBIN = "round_robin"  # Distribute evenly
    RANDOM = "random"  # Random selection
    WEIGHTED = "weighted"  # Weighted distribution
    LEAST_CALLS = "least_calls"  # Route to least busy
    PRIORITY = "priority"  # By priority value
    FAILOVER = "failover"  # Primary with backups


class RoutingConditionType(str, Enum):
    """Types of routing conditions."""
    TIME_OF_DAY = "time_of_day"
    DAY_OF_WEEK = "day_of_week"
    CALLER_NUMBER = "caller_number"
    CALLED_NUMBER = "called_number"
    CALL_DIRECTION = "call_direction"
    CALLER_REGION = "caller_region"
    CUSTOM = "custom"


class RouteAction(str, Enum):
    """Actions for routing."""
    ROUTE_TO_AGENT = "route_to_agent"
    ROUTE_TO_NUMBER = "route_to_number"
    ROUTE_TO_QUEUE = "route_to_queue"
    ROUTE_TO_VOICEMAIL = "route_to_voicemail"
    PLAY_MESSAGE = "play_message"
    TRANSFER = "transfer"
    REJECT = "reject"
    HANGUP = "hangup"


@dataclass
class RouteDestination:
    """Routing destination."""

    id: str = ""
    type: RouteAction = RouteAction.ROUTE_TO_AGENT

    # Destination details
    agent_id: Optional[str] = None
    phone_number: Optional[str] = None
    queue_name: Optional[str] = None
    voicemail_box: Optional[str] = None
    message_url: Optional[str] = None
    message_text: Optional[str] = None

    # Weighting
    weight: int = 1
    priority: int = 0

    # Capacity
    max_concurrent: int = 100
    current_calls: int = 0

    # Availability
    available: bool = True
    available_start: Optional[time] = None
    available_end: Optional[time] = None

    # Stats
    total_calls: int = 0
    successful_calls: int = 0

    def is_available(self) -> bool:
        """Check if destination is currently available."""
        if not self.available:
            return False

        if self.current_calls >= self.max_concurrent:
            return False

        if self.available_start and self.available_end:
            now = datetime.now().time()
            if not (self.available_start <= now <= self.available_end):
                return False

        return True


@dataclass
class RoutingCondition:
    """Condition for routing rule."""

    type: RoutingConditionType = RoutingConditionType.CUSTOM
    operator: str = "equals"  # equals, contains, matches, in_range, not_equals

    # Condition values
    value: Any = None
    values: List[Any] = field(default_factory=list)

    # Time conditions
    start_time: Optional[time] = None
    end_time: Optional[time] = None
    days_of_week: List[int] = field(default_factory=list)  # 0=Monday

    # Custom function
    custom_function: Optional[Callable[[Call], bool]] = None

    def evaluate(self, call: Call) -> bool:
        """Evaluate condition against a call."""
        if self.type == RoutingConditionType.TIME_OF_DAY:
            return self._check_time_of_day()

        elif self.type == RoutingConditionType.DAY_OF_WEEK:
            return self._check_day_of_week()

        elif self.type == RoutingConditionType.CALLER_NUMBER:
            return self._check_value(call.from_number)

        elif self.type == RoutingConditionType.CALLED_NUMBER:
            return self._check_value(call.to_number)

        elif self.type == RoutingConditionType.CALL_DIRECTION:
            return call.direction.value == self.value

        elif self.type == RoutingConditionType.CUSTOM:
            if self.custom_function:
                return self.custom_function(call)

        return True

    def _check_time_of_day(self) -> bool:
        """Check if current time is in range."""
        if self.start_time and self.end_time:
            now = datetime.now().time()
            return self.start_time <= now <= self.end_time
        return True

    def _check_day_of_week(self) -> bool:
        """Check if current day is in list."""
        if self.days_of_week:
            today = datetime.now().weekday()
            return today in self.days_of_week
        return True

    def _check_value(self, actual: str) -> bool:
        """Check value against condition."""
        if self.operator == "equals":
            return actual == self.value

        elif self.operator == "not_equals":
            return actual != self.value

        elif self.operator == "contains":
            return self.value in actual

        elif self.operator == "starts_with":
            return actual.startswith(self.value)

        elif self.operator == "ends_with":
            return actual.endswith(self.value)

        elif self.operator == "matches":
            return bool(re.match(self.value, actual))

        elif self.operator == "in":
            return actual in self.values

        elif self.operator == "not_in":
            return actual not in self.values

        return False


@dataclass
class RoutingRule:
    """Routing rule with conditions and destinations."""

    id: str = ""
    name: str = ""
    description: str = ""

    # Conditions (all must match)
    conditions: List[RoutingCondition] = field(default_factory=list)

    # Destinations
    destinations: List[RouteDestination] = field(default_factory=list)
    strategy: RoutingStrategy = RoutingStrategy.SEQUENTIAL

    # Priority (higher = evaluated first)
    priority: int = 0

    # Enabled
    enabled: bool = True

    # Fallback
    fallback_action: RouteAction = RouteAction.ROUTE_TO_VOICEMAIL
    fallback_destination: Optional[RouteDestination] = None

    # Timing
    timeout_seconds: int = 30
    retry_count: int = 2
    retry_delay_seconds: int = 5

    def matches(self, call: Call) -> bool:
        """Check if rule matches the call."""
        if not self.enabled:
            return False

        return all(condition.evaluate(call) for condition in self.conditions)


@dataclass
class RoutingConfig:
    """Routing configuration."""

    # Default strategy
    default_strategy: RoutingStrategy = RoutingStrategy.SEQUENTIAL

    # Rules
    rules: List[RoutingRule] = field(default_factory=list)

    # Default destinations
    default_agent_destination: Optional[RouteDestination] = None
    default_voicemail_destination: Optional[RouteDestination] = None

    # Timeouts
    default_timeout_seconds: int = 30
    max_ring_time_seconds: int = 60

    # After hours
    after_hours_action: RouteAction = RouteAction.ROUTE_TO_VOICEMAIL
    after_hours_destination: Optional[RouteDestination] = None

    # Business hours
    business_hours_start: time = time(9, 0)
    business_hours_end: time = time(17, 0)
    business_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri


class CallRouter:
    """
    Routes incoming calls based on rules and conditions.

    Supports multiple routing strategies and load balancing.
    """

    def __init__(self, config: Optional[RoutingConfig] = None):
        """
        Initialize call router.

        Args:
            config: Routing configuration
        """
        self.config = config or RoutingConfig()
        self._destination_stats: Dict[str, Dict] = {}
        self._round_robin_index: int = 0

    def route(self, call: Call) -> Optional[RouteDestination]:
        """
        Route a call to a destination.

        Args:
            call: Call to route

        Returns:
            Selected destination or None
        """
        # Check business hours first
        if not self._is_business_hours():
            logger.info(f"Routing call {call.id} to after-hours")
            return self._get_after_hours_destination()

        # Find matching rule
        matched_rule = self._find_matching_rule(call)

        if matched_rule:
            destination = self._select_destination(matched_rule)
            if destination:
                logger.info(
                    f"Routing call {call.id} via rule '{matched_rule.name}' "
                    f"to {destination.type.value}"
                )
                return destination

        # No rule matched, use default
        logger.info(f"Routing call {call.id} to default destination")
        return self.config.default_agent_destination

    def route_with_fallback(
        self,
        call: Call,
    ) -> List[RouteDestination]:
        """
        Get routing destinations with fallbacks.

        Args:
            call: Call to route

        Returns:
            List of destinations to try in order
        """
        destinations = []

        # Check business hours
        if not self._is_business_hours():
            after_hours = self._get_after_hours_destination()
            if after_hours:
                destinations.append(after_hours)
            return destinations

        # Find matching rule
        matched_rule = self._find_matching_rule(call)

        if matched_rule:
            # Get primary destinations based on strategy
            primary_destinations = self._get_ordered_destinations(matched_rule)
            destinations.extend(primary_destinations)

            # Add fallback if defined
            if matched_rule.fallback_destination:
                destinations.append(matched_rule.fallback_destination)

        # Add global fallback
        if self.config.default_voicemail_destination:
            destinations.append(self.config.default_voicemail_destination)

        return destinations

    def add_rule(self, rule: RoutingRule) -> None:
        """Add a routing rule."""
        self.config.rules.append(rule)
        # Sort by priority
        self.config.rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a routing rule."""
        for i, rule in enumerate(self.config.rules):
            if rule.id == rule_id:
                self.config.rules.pop(i)
                return True
        return False

    def update_destination_stats(
        self,
        destination_id: str,
        call_started: bool = False,
        call_ended: bool = False,
        successful: bool = True,
    ) -> None:
        """Update statistics for a destination."""
        if destination_id not in self._destination_stats:
            self._destination_stats[destination_id] = {
                "current_calls": 0,
                "total_calls": 0,
                "successful_calls": 0,
            }

        stats = self._destination_stats[destination_id]

        if call_started:
            stats["current_calls"] += 1
            stats["total_calls"] += 1

        if call_ended:
            stats["current_calls"] = max(0, stats["current_calls"] - 1)
            if successful:
                stats["successful_calls"] += 1

    def _find_matching_rule(self, call: Call) -> Optional[RoutingRule]:
        """Find first matching rule for a call."""
        for rule in self.config.rules:
            if rule.matches(call):
                return rule
        return None

    def _select_destination(
        self,
        rule: RoutingRule,
    ) -> Optional[RouteDestination]:
        """Select a destination based on rule strategy."""
        available_destinations = [
            d for d in rule.destinations if d.is_available()
        ]

        if not available_destinations:
            return rule.fallback_destination

        strategy = rule.strategy

        if strategy == RoutingStrategy.SEQUENTIAL:
            return available_destinations[0]

        elif strategy == RoutingStrategy.ROUND_ROBIN:
            idx = self._round_robin_index % len(available_destinations)
            self._round_robin_index += 1
            return available_destinations[idx]

        elif strategy == RoutingStrategy.RANDOM:
            import random
            return random.choice(available_destinations)

        elif strategy == RoutingStrategy.WEIGHTED:
            return self._weighted_select(available_destinations)

        elif strategy == RoutingStrategy.LEAST_CALLS:
            return min(
                available_destinations,
                key=lambda d: d.current_calls
            )

        elif strategy == RoutingStrategy.PRIORITY:
            return max(
                available_destinations,
                key=lambda d: d.priority
            )

        elif strategy == RoutingStrategy.FAILOVER:
            # Return first available by priority
            sorted_dest = sorted(
                available_destinations,
                key=lambda d: d.priority,
                reverse=True
            )
            return sorted_dest[0] if sorted_dest else None

        return available_destinations[0]

    def _get_ordered_destinations(
        self,
        rule: RoutingRule,
    ) -> List[RouteDestination]:
        """Get destinations in order based on strategy."""
        available = [d for d in rule.destinations if d.is_available()]

        if not available:
            return []

        strategy = rule.strategy

        if strategy == RoutingStrategy.SEQUENTIAL:
            return available

        elif strategy == RoutingStrategy.ROUND_ROBIN:
            # Rotate list
            idx = self._round_robin_index % len(available)
            self._round_robin_index += 1
            return available[idx:] + available[:idx]

        elif strategy == RoutingStrategy.RANDOM:
            import random
            shuffled = list(available)
            random.shuffle(shuffled)
            return shuffled

        elif strategy == RoutingStrategy.PRIORITY:
            return sorted(available, key=lambda d: d.priority, reverse=True)

        elif strategy == RoutingStrategy.LEAST_CALLS:
            return sorted(available, key=lambda d: d.current_calls)

        return available

    def _weighted_select(
        self,
        destinations: List[RouteDestination],
    ) -> RouteDestination:
        """Select destination by weight."""
        import random

        total_weight = sum(d.weight for d in destinations)
        r = random.uniform(0, total_weight)

        cumulative = 0
        for dest in destinations:
            cumulative += dest.weight
            if r <= cumulative:
                return dest

        return destinations[-1]

    def _is_business_hours(self) -> bool:
        """Check if currently within business hours."""
        now = datetime.now()
        current_time = now.time()
        current_day = now.weekday()

        if current_day not in self.config.business_days:
            return False

        if not (self.config.business_hours_start <= current_time <= self.config.business_hours_end):
            return False

        return True

    def _get_after_hours_destination(self) -> Optional[RouteDestination]:
        """Get after-hours destination."""
        if self.config.after_hours_destination:
            return self.config.after_hours_destination

        # Create default voicemail destination
        if self.config.default_voicemail_destination:
            return self.config.default_voicemail_destination

        return RouteDestination(
            type=RouteAction.ROUTE_TO_VOICEMAIL,
        )


# Convenience functions for creating routing rules

def create_time_based_rule(
    name: str,
    start_time: time,
    end_time: time,
    destination: RouteDestination,
    days: Optional[List[int]] = None,
) -> RoutingRule:
    """Create a time-based routing rule."""
    conditions = [
        RoutingCondition(
            type=RoutingConditionType.TIME_OF_DAY,
            start_time=start_time,
            end_time=end_time,
        )
    ]

    if days:
        conditions.append(
            RoutingCondition(
                type=RoutingConditionType.DAY_OF_WEEK,
                days_of_week=days,
            )
        )

    return RoutingRule(
        name=name,
        conditions=conditions,
        destinations=[destination],
    )


def create_number_based_rule(
    name: str,
    pattern: str,
    destination: RouteDestination,
    is_caller: bool = True,
) -> RoutingRule:
    """Create a number pattern routing rule."""
    condition_type = (
        RoutingConditionType.CALLER_NUMBER
        if is_caller
        else RoutingConditionType.CALLED_NUMBER
    )

    return RoutingRule(
        name=name,
        conditions=[
            RoutingCondition(
                type=condition_type,
                operator="matches",
                value=pattern,
            )
        ],
        destinations=[destination],
    )


__all__ = [
    # Enums
    "RoutingStrategy",
    "RoutingConditionType",
    "RouteAction",
    # Data classes
    "RouteDestination",
    "RoutingCondition",
    "RoutingRule",
    "RoutingConfig",
    # Classes
    "CallRouter",
    # Functions
    "create_time_based_rule",
    "create_number_based_rule",
]
