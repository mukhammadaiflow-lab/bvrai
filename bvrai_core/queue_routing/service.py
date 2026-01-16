"""
Queue & Routing Service Module

This module provides comprehensive queue management, routing, and analytics
services for voice agent call handling.
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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

from .base import (
    # Enums
    AgentStatus,
    CallOutcome,
    OverflowAction,
    QueuePriority,
    QueueStatus,
    RoutingStrategy,
    SkillLevel,
    # Types
    AgentMetrics,
    AgentSkill,
    CallbackRequest,
    CallQueue,
    QueueAgent,
    QueueConfig,
    QueueEntry,
    QueueMetrics,
    RoutingDecision,
    Skill,
    SkillRequirement,
    # Exceptions
    AgentNotFoundError,
    NoAgentAvailableError,
    QueueClosedError,
    QueueFullError,
    QueueNotFoundError,
    QueueRoutingError,
    SkillMismatchError,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Skill Manager
# =============================================================================


class SkillManager:
    """Manages skills for the organization."""

    def __init__(self):
        self._skills: Dict[str, Skill] = {}
        self._skills_by_org: Dict[str, Set[str]] = defaultdict(set)

    async def create_skill(
        self,
        organization_id: str,
        name: str,
        category: str = "general",
        description: str = "",
    ) -> Skill:
        """Create a new skill."""
        skill = Skill(
            id=f"skill_{uuid.uuid4().hex[:16]}",
            name=name,
            category=category,
            description=description,
        )

        self._skills[skill.id] = skill
        self._skills_by_org[organization_id].add(skill.id)

        logger.info(f"Created skill {skill.id}: {name}")

        return skill

    async def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Get skill by ID."""
        return self._skills.get(skill_id)

    async def list_skills(
        self,
        organization_id: str,
        category: Optional[str] = None,
        active_only: bool = True,
    ) -> List[Skill]:
        """List skills for organization."""
        skill_ids = self._skills_by_org.get(organization_id, set())
        skills = []

        for skill_id in skill_ids:
            skill = self._skills.get(skill_id)
            if not skill:
                continue
            if active_only and not skill.is_active:
                continue
            if category and skill.category != category:
                continue
            skills.append(skill)

        return sorted(skills, key=lambda s: s.name)

    async def update_skill(
        self,
        skill_id: str,
        name: Optional[str] = None,
        category: Optional[str] = None,
        description: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> Skill:
        """Update a skill."""
        skill = self._skills.get(skill_id)
        if not skill:
            raise QueueRoutingError(f"Skill {skill_id} not found")

        if name:
            skill.name = name
        if category:
            skill.category = category
        if description is not None:
            skill.description = description
        if is_active is not None:
            skill.is_active = is_active

        return skill

    async def delete_skill(self, skill_id: str) -> bool:
        """Delete a skill."""
        skill = self._skills.get(skill_id)
        if not skill:
            return False

        del self._skills[skill_id]
        for org_skills in self._skills_by_org.values():
            org_skills.discard(skill_id)

        return True


# =============================================================================
# Agent Manager
# =============================================================================


class AgentManager:
    """
    Manages queue agents and their availability.

    Features:
    - Agent CRUD operations
    - Status management
    - Skill assignment
    - Capacity tracking
    """

    def __init__(self):
        self._agents: Dict[str, QueueAgent] = {}
        self._agents_by_org: Dict[str, Set[str]] = defaultdict(set)
        self._agents_by_queue: Dict[str, Set[str]] = defaultdict(set)
        self._status_callbacks: List[Callable[[QueueAgent, AgentStatus], None]] = []

    async def create_agent(
        self,
        organization_id: str,
        name: str,
        email: str,
        skills: Optional[List[AgentSkill]] = None,
        queue_ids: Optional[List[str]] = None,
        max_concurrent_calls: int = 1,
        is_ai_agent: bool = False,
    ) -> QueueAgent:
        """Create a new queue agent."""
        agent = QueueAgent(
            id=f"qagent_{uuid.uuid4().hex[:20]}",
            organization_id=organization_id,
            name=name,
            email=email,
            skills=skills or [],
            queue_ids=queue_ids or [],
            max_concurrent_calls=max_concurrent_calls,
            is_ai_agent=is_ai_agent,
        )

        self._agents[agent.id] = agent
        self._agents_by_org[organization_id].add(agent.id)

        for queue_id in agent.queue_ids:
            self._agents_by_queue[queue_id].add(agent.id)

        logger.info(f"Created queue agent {agent.id}: {name}")

        return agent

    async def get_agent(self, agent_id: str) -> Optional[QueueAgent]:
        """Get agent by ID."""
        return self._agents.get(agent_id)

    async def list_agents(
        self,
        organization_id: str,
        queue_id: Optional[str] = None,
        status: Optional[AgentStatus] = None,
        available_only: bool = False,
    ) -> List[QueueAgent]:
        """List agents with optional filters."""
        if queue_id:
            agent_ids = self._agents_by_queue.get(queue_id, set())
        else:
            agent_ids = self._agents_by_org.get(organization_id, set())

        agents = []
        for agent_id in agent_ids:
            agent = self._agents.get(agent_id)
            if not agent:
                continue
            if status and agent.status != status:
                continue
            if available_only and not agent.is_available:
                continue
            agents.append(agent)

        return agents

    async def get_available_agents(
        self,
        queue_id: str,
        skill_requirements: Optional[List[SkillRequirement]] = None,
    ) -> List[Tuple[QueueAgent, float]]:
        """
        Get available agents for a queue with skill matching.

        Returns:
            List of (agent, match_score) tuples sorted by score
        """
        agent_ids = self._agents_by_queue.get(queue_id, set())
        available = []

        for agent_id in agent_ids:
            agent = self._agents.get(agent_id)
            if not agent or not agent.is_available:
                continue

            if skill_requirements:
                meets_reqs, score = agent.meets_requirements(skill_requirements)
                if meets_reqs:
                    available.append((agent, score))
            else:
                available.append((agent, 1.0))

        return sorted(available, key=lambda x: x[1], reverse=True)

    async def set_agent_status(
        self,
        agent_id: str,
        status: AgentStatus,
        reason: str = "",
    ) -> QueueAgent:
        """Update agent status."""
        agent = await self.get_agent(agent_id)
        if not agent:
            raise AgentNotFoundError(f"Agent {agent_id} not found")

        old_status = agent.status
        agent.status = status
        agent.status_reason = reason
        agent.status_changed_at = datetime.utcnow()
        agent.updated_at = datetime.utcnow()

        if status == AgentStatus.AVAILABLE:
            agent.last_available_at = datetime.utcnow()

        # Trigger callbacks
        for callback in self._status_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(agent, old_status)
                else:
                    callback(agent, old_status)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")

        logger.info(f"Agent {agent_id} status changed: {old_status.value} -> {status.value}")

        return agent

    async def assign_call(self, agent_id: str) -> bool:
        """Assign a call to an agent."""
        agent = await self.get_agent(agent_id)
        if not agent:
            return False

        if not agent.is_available:
            return False

        agent.current_calls += 1
        agent.total_calls_handled += 1
        agent.last_call_at = datetime.utcnow()
        agent.updated_at = datetime.utcnow()

        # Update status if at capacity
        if agent.current_calls >= agent.max_concurrent_calls:
            agent.status = AgentStatus.BUSY

        return True

    async def release_call(
        self,
        agent_id: str,
        handle_time_seconds: float = 0.0,
        wrap_up_time_seconds: float = 30.0,
    ) -> bool:
        """Release a call from an agent."""
        agent = await self.get_agent(agent_id)
        if not agent:
            return False

        agent.current_calls = max(0, agent.current_calls - 1)
        agent.updated_at = datetime.utcnow()

        # Update average handle time
        if handle_time_seconds > 0:
            total_time = agent.avg_handle_time_seconds * (agent.total_calls_handled - 1)
            total_time += handle_time_seconds
            agent.avg_handle_time_seconds = total_time / agent.total_calls_handled
            agent.total_handle_time_seconds += handle_time_seconds

        # Enter wrap-up if configured
        if wrap_up_time_seconds > 0:
            agent.status = AgentStatus.WRAP_UP
            # Schedule return to available
            asyncio.create_task(
                self._return_to_available_after_wrap(agent_id, wrap_up_time_seconds)
            )
        elif agent.current_calls == 0:
            agent.status = AgentStatus.AVAILABLE

        return True

    async def _return_to_available_after_wrap(
        self,
        agent_id: str,
        wrap_up_seconds: float,
    ) -> None:
        """Return agent to available status after wrap-up."""
        await asyncio.sleep(wrap_up_seconds)
        agent = await self.get_agent(agent_id)
        if agent and agent.status == AgentStatus.WRAP_UP and agent.current_calls == 0:
            agent.status = AgentStatus.AVAILABLE
            agent.last_available_at = datetime.utcnow()

    async def add_skill(
        self,
        agent_id: str,
        skill_id: str,
        skill_name: str,
        level: SkillLevel = SkillLevel.INTERMEDIATE,
        certified: bool = False,
    ) -> QueueAgent:
        """Add a skill to an agent."""
        agent = await self.get_agent(agent_id)
        if not agent:
            raise AgentNotFoundError(f"Agent {agent_id} not found")

        # Check if skill already exists
        for existing in agent.skills:
            if existing.skill_id == skill_id:
                existing.level = level
                existing.certified = certified
                if certified:
                    existing.certified_at = datetime.utcnow()
                return agent

        # Add new skill
        agent.skills.append(AgentSkill(
            skill_id=skill_id,
            skill_name=skill_name,
            level=level,
            certified=certified,
            certified_at=datetime.utcnow() if certified else None,
        ))
        agent.updated_at = datetime.utcnow()

        return agent

    async def remove_skill(self, agent_id: str, skill_id: str) -> QueueAgent:
        """Remove a skill from an agent."""
        agent = await self.get_agent(agent_id)
        if not agent:
            raise AgentNotFoundError(f"Agent {agent_id} not found")

        agent.skills = [s for s in agent.skills if s.skill_id != skill_id]
        agent.updated_at = datetime.utcnow()

        return agent

    async def assign_to_queue(self, agent_id: str, queue_id: str) -> QueueAgent:
        """Assign an agent to a queue."""
        agent = await self.get_agent(agent_id)
        if not agent:
            raise AgentNotFoundError(f"Agent {agent_id} not found")

        if queue_id not in agent.queue_ids:
            agent.queue_ids.append(queue_id)
            self._agents_by_queue[queue_id].add(agent_id)
            agent.updated_at = datetime.utcnow()

        return agent

    async def remove_from_queue(self, agent_id: str, queue_id: str) -> QueueAgent:
        """Remove an agent from a queue."""
        agent = await self.get_agent(agent_id)
        if not agent:
            raise AgentNotFoundError(f"Agent {agent_id} not found")

        if queue_id in agent.queue_ids:
            agent.queue_ids.remove(queue_id)
            self._agents_by_queue[queue_id].discard(agent_id)
            agent.updated_at = datetime.utcnow()

        return agent

    def add_status_callback(
        self,
        callback: Callable[[QueueAgent, AgentStatus], None],
    ) -> None:
        """Add callback for agent status changes."""
        self._status_callbacks.append(callback)


# =============================================================================
# Queue Manager
# =============================================================================


class QueueManager:
    """
    Manages call queues and their entries.

    Features:
    - Queue CRUD operations
    - Entry management
    - Position tracking
    - Overflow handling
    """

    def __init__(self):
        self._queues: Dict[str, CallQueue] = {}
        self._queues_by_org: Dict[str, Set[str]] = defaultdict(set)
        self._entries: Dict[str, QueueEntry] = {}
        self._entries_by_queue: Dict[str, List[str]] = defaultdict(list)

    async def create_queue(
        self,
        organization_id: str,
        name: str,
        config: Optional[QueueConfig] = None,
        skill_requirements: Optional[List[SkillRequirement]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> CallQueue:
        """Create a new call queue."""
        queue = CallQueue(
            id=f"queue_{uuid.uuid4().hex[:20]}",
            organization_id=organization_id,
            name=name,
            config=config or QueueConfig(),
            skill_requirements=skill_requirements or [],
            description=description,
            tags=tags or [],
        )

        self._queues[queue.id] = queue
        self._queues_by_org[organization_id].add(queue.id)

        logger.info(f"Created queue {queue.id}: {name}")

        return queue

    async def get_queue(self, queue_id: str) -> Optional[CallQueue]:
        """Get queue by ID."""
        return self._queues.get(queue_id)

    async def list_queues(
        self,
        organization_id: str,
        status: Optional[QueueStatus] = None,
        tag: Optional[str] = None,
    ) -> List[CallQueue]:
        """List queues for organization."""
        queue_ids = self._queues_by_org.get(organization_id, set())
        queues = []

        for queue_id in queue_ids:
            queue = self._queues.get(queue_id)
            if not queue:
                continue
            if status and queue.status != status:
                continue
            if tag and tag not in queue.tags:
                continue
            queues.append(queue)

        return sorted(queues, key=lambda q: q.name)

    async def update_queue(
        self,
        queue_id: str,
        name: Optional[str] = None,
        config: Optional[QueueConfig] = None,
        skill_requirements: Optional[List[SkillRequirement]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> CallQueue:
        """Update queue configuration."""
        queue = await self.get_queue(queue_id)
        if not queue:
            raise QueueNotFoundError(f"Queue {queue_id} not found")

        if name:
            queue.name = name
        if config:
            queue.config = config
        if skill_requirements is not None:
            queue.skill_requirements = skill_requirements
        if description is not None:
            queue.description = description
        if tags is not None:
            queue.tags = tags

        queue.updated_at = datetime.utcnow()

        return queue

    async def set_queue_status(
        self,
        queue_id: str,
        status: QueueStatus,
        reason: str = "",
    ) -> CallQueue:
        """Set queue status."""
        queue = await self.get_queue(queue_id)
        if not queue:
            raise QueueNotFoundError(f"Queue {queue_id} not found")

        queue.status = status
        queue.status_reason = reason
        queue.updated_at = datetime.utcnow()

        logger.info(f"Queue {queue_id} status set to {status.value}")

        return queue

    async def add_entry(
        self,
        queue_id: str,
        call_id: str,
        caller_phone: str,
        caller_name: Optional[str] = None,
        caller_id: Optional[str] = None,
        priority: QueuePriority = QueuePriority.NORMAL,
        skill_requirements: Optional[List[SkillRequirement]] = None,
        is_vip: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> QueueEntry:
        """Add a call to the queue."""
        queue = await self.get_queue(queue_id)
        if not queue:
            raise QueueNotFoundError(f"Queue {queue_id} not found")

        if not queue.is_accepting_calls:
            if queue.status == QueueStatus.CLOSED:
                raise QueueClosedError(f"Queue {queue_id} is closed")
            if queue.is_at_capacity:
                raise QueueFullError(f"Queue {queue_id} is at capacity")
            raise QueueRoutingError(f"Queue {queue_id} is not accepting calls")

        # Calculate position
        position = len(self._entries_by_queue[queue_id]) + 1

        # Apply VIP boost
        if is_vip and queue.config.allow_priority_override:
            priority_order = [QueuePriority.LOW, QueuePriority.NORMAL, QueuePriority.HIGH, QueuePriority.CRITICAL]
            current_idx = priority_order.index(priority)
            boosted_idx = min(current_idx + queue.config.vip_priority_boost, len(priority_order) - 1)
            priority = priority_order[boosted_idx]

        # Create entry
        entry = QueueEntry(
            id=f"qentry_{uuid.uuid4().hex[:18]}",
            queue_id=queue_id,
            organization_id=queue.organization_id,
            call_id=call_id,
            caller_phone=caller_phone,
            caller_name=caller_name,
            caller_id=caller_id,
            priority=priority,
            skill_requirements=skill_requirements or queue.skill_requirements,
            is_vip=is_vip,
            position=position,
            max_wait_until=datetime.utcnow() + timedelta(
                seconds=queue.config.max_wait_time_seconds
            ),
            context=context or {},
        )

        # Store entry
        self._entries[entry.id] = entry
        self._entries_by_queue[queue_id].append(entry.id)

        # Update queue stats
        queue.current_size += 1
        queue.total_calls_today += 1

        # Recalculate positions based on priority
        await self._reorder_queue(queue_id)

        logger.info(f"Added entry {entry.id} to queue {queue_id} at position {entry.position}")

        return entry

    async def get_entry(self, entry_id: str) -> Optional[QueueEntry]:
        """Get queue entry by ID."""
        return self._entries.get(entry_id)

    async def get_entries(
        self,
        queue_id: str,
        waiting_only: bool = True,
    ) -> List[QueueEntry]:
        """Get all entries in a queue."""
        entry_ids = self._entries_by_queue.get(queue_id, [])
        entries = []

        for entry_id in entry_ids:
            entry = self._entries.get(entry_id)
            if entry:
                if waiting_only and not entry.is_waiting:
                    continue
                entries.append(entry)

        return sorted(entries, key=lambda e: e.position)

    async def get_next_entry(self, queue_id: str) -> Optional[QueueEntry]:
        """Get the next entry to be processed."""
        entries = await self.get_entries(queue_id, waiting_only=True)
        if not entries:
            return None
        return entries[0]

    async def _reorder_queue(self, queue_id: str) -> None:
        """Reorder queue entries by priority score."""
        entries = await self.get_entries(queue_id, waiting_only=True)

        # Sort by priority score (higher = first)
        entries.sort(key=lambda e: e.priority_score, reverse=True)

        # Update positions
        for i, entry in enumerate(entries):
            entry.position = i + 1

    async def complete_entry(
        self,
        entry_id: str,
        outcome: CallOutcome,
        agent_id: Optional[str] = None,
    ) -> QueueEntry:
        """Mark an entry as completed."""
        entry = await self.get_entry(entry_id)
        if not entry:
            raise QueueRoutingError(f"Entry {entry_id} not found")

        queue = await self.get_queue(entry.queue_id)

        entry.outcome = outcome
        entry.outcome_at = datetime.utcnow()
        if agent_id:
            entry.assigned_agent_id = agent_id
            entry.assigned_at = datetime.utcnow()

        # Update queue stats
        if queue:
            queue.current_size = max(0, queue.current_size - 1)

            wait_time = entry.wait_seconds
            if outcome == CallOutcome.ANSWERED:
                queue.calls_answered_today += 1
                # Update average wait time
                total_wait = queue.avg_wait_time_today_seconds * (queue.calls_answered_today - 1)
                total_wait += wait_time
                queue.avg_wait_time_today_seconds = total_wait / queue.calls_answered_today

                # Update SLA
                if wait_time <= queue.config.target_answer_time_seconds:
                    queue.sla_current_percent = (
                        (queue.sla_current_percent * (queue.calls_answered_today - 1) + 100)
                        / queue.calls_answered_today
                    )
                else:
                    queue.sla_current_percent = (
                        queue.sla_current_percent * (queue.calls_answered_today - 1)
                        / queue.calls_answered_today
                    )
            elif outcome == CallOutcome.ABANDONED:
                queue.calls_abandoned_today += 1

            # Recalculate positions
            await self._reorder_queue(entry.queue_id)

        # Remove from entries list
        if entry.queue_id in self._entries_by_queue:
            self._entries_by_queue[entry.queue_id] = [
                eid for eid in self._entries_by_queue[entry.queue_id]
                if eid != entry_id
            ]

        logger.info(f"Entry {entry_id} completed with outcome {outcome.value}")

        return entry

    async def request_callback(
        self,
        entry_id: str,
        callback_phone: Optional[str] = None,
    ) -> QueueEntry:
        """Request a callback for a queue entry."""
        entry = await self.get_entry(entry_id)
        if not entry:
            raise QueueRoutingError(f"Entry {entry_id} not found")

        entry.callback_requested = True
        entry.callback_phone = callback_phone or entry.caller_phone
        entry.callback_scheduled_at = datetime.utcnow()

        return entry

    async def estimate_wait_time(self, queue_id: str) -> float:
        """Estimate wait time for new entries in seconds."""
        queue = await self.get_queue(queue_id)
        if not queue:
            return 0.0

        # Base estimate on current queue size and average handle time
        # This is simplified - real implementation would use historical data
        waiting_count = queue.current_size
        avg_handle = queue.avg_wait_time_today_seconds or 120.0  # Default 2 min

        return waiting_count * avg_handle

    async def get_longest_waiting_entry(self, queue_id: str) -> Optional[QueueEntry]:
        """Get the entry that has been waiting longest."""
        entries = await self.get_entries(queue_id, waiting_only=True)
        if not entries:
            return None
        return max(entries, key=lambda e: e.wait_seconds)


# =============================================================================
# Routing Engine
# =============================================================================


class RoutingStrategy(ABC):
    """Abstract base class for routing strategies."""

    @abstractmethod
    async def select_agent(
        self,
        entry: QueueEntry,
        available_agents: List[Tuple[QueueAgent, float]],
    ) -> Optional[Tuple[QueueAgent, float, str]]:
        """
        Select an agent for the entry.

        Returns:
            Tuple of (agent, score, reason) or None if no agent selected
        """
        pass


class RoundRobinStrategy(RoutingStrategy):
    """Round robin routing strategy."""

    def __init__(self):
        self._last_agent: Dict[str, int] = {}  # queue_id -> last index

    async def select_agent(
        self,
        entry: QueueEntry,
        available_agents: List[Tuple[QueueAgent, float]],
    ) -> Optional[Tuple[QueueAgent, float, str]]:
        if not available_agents:
            return None

        last_idx = self._last_agent.get(entry.queue_id, -1)
        next_idx = (last_idx + 1) % len(available_agents)
        self._last_agent[entry.queue_id] = next_idx

        agent, score = available_agents[next_idx]
        return agent, score, "Round robin selection"


class LeastBusyStrategy(RoutingStrategy):
    """Route to agent with lowest current call count."""

    async def select_agent(
        self,
        entry: QueueEntry,
        available_agents: List[Tuple[QueueAgent, float]],
    ) -> Optional[Tuple[QueueAgent, float, str]]:
        if not available_agents:
            return None

        # Sort by current calls (ascending)
        sorted_agents = sorted(
            available_agents,
            key=lambda x: (x[0].current_calls, -x[1])  # Least busy, then highest score
        )

        agent, score = sorted_agents[0]
        return agent, score, f"Least busy agent ({agent.current_calls} calls)"


class LongestIdleStrategy(RoutingStrategy):
    """Route to agent that has been idle longest."""

    async def select_agent(
        self,
        entry: QueueEntry,
        available_agents: List[Tuple[QueueAgent, float]],
    ) -> Optional[Tuple[QueueAgent, float, str]]:
        if not available_agents:
            return None

        # Sort by idle time (descending)
        sorted_agents = sorted(
            available_agents,
            key=lambda x: (x[0].idle_time, x[1]),  # Longest idle, then highest score
            reverse=True,
        )

        agent, score = sorted_agents[0]
        idle_mins = agent.idle_time.total_seconds() / 60
        return agent, score, f"Longest idle agent ({idle_mins:.1f} min)"


class SkillsBasedStrategy(RoutingStrategy):
    """Route based on skill match score."""

    async def select_agent(
        self,
        entry: QueueEntry,
        available_agents: List[Tuple[QueueAgent, float]],
    ) -> Optional[Tuple[QueueAgent, float, str]]:
        if not available_agents:
            return None

        # Already sorted by skill score
        agent, score = available_agents[0]
        return agent, score, f"Best skill match (score: {score:.2f})"


class WeightedStrategy(RoutingStrategy):
    """Weighted random selection based on agent weights."""

    async def select_agent(
        self,
        entry: QueueEntry,
        available_agents: List[Tuple[QueueAgent, float]],
    ) -> Optional[Tuple[QueueAgent, float, str]]:
        if not available_agents:
            return None

        import random

        # Calculate weighted selection
        total_weight = sum(a[0].weight for a in available_agents)
        r = random.uniform(0, total_weight)

        cumulative = 0.0
        for agent, score in available_agents:
            cumulative += agent.weight
            if r <= cumulative:
                return agent, score, f"Weighted selection (weight: {agent.weight})"

        # Fallback to first
        agent, score = available_agents[0]
        return agent, score, "Weighted selection fallback"


class StickyStrategy(RoutingStrategy):
    """Try to route to the same agent as previous interaction."""

    async def select_agent(
        self,
        entry: QueueEntry,
        available_agents: List[Tuple[QueueAgent, float]],
    ) -> Optional[Tuple[QueueAgent, float, str]]:
        if not available_agents:
            return None

        # Check for preferred agent
        if entry.preferred_agent_id:
            for agent, score in available_agents:
                if agent.id == entry.preferred_agent_id:
                    return agent, score, "Sticky routing to preferred agent"

        # Fall back to first available (typically skills-sorted)
        agent, score = available_agents[0]
        return agent, score, "Preferred agent unavailable, best match selected"


class RoutingEngine:
    """
    Core routing engine for matching calls to agents.

    Features:
    - Multiple routing strategies
    - Skill-based matching
    - Overflow handling
    - Routing decision tracking
    """

    def __init__(self, agent_manager: AgentManager):
        self._agent_manager = agent_manager
        self._strategies: Dict[str, RoutingStrategy] = {
            "round_robin": RoundRobinStrategy(),
            "least_busy": LeastBusyStrategy(),
            "longest_idle": LongestIdleStrategy(),
            "skills_based": SkillsBasedStrategy(),
            "weighted": WeightedStrategy(),
            "sticky": StickyStrategy(),
        }
        self._decisions: List[RoutingDecision] = []

    async def route(
        self,
        entry: QueueEntry,
        queue: CallQueue,
        force_strategy: Optional[str] = None,
    ) -> RoutingDecision:
        """
        Route a queue entry to an agent.

        Returns:
            RoutingDecision with routing result
        """
        strategy_name = force_strategy or queue.config.routing_strategy.value

        # Get available agents
        available_agents = await self._agent_manager.get_available_agents(
            queue.id,
            skill_requirements=entry.skill_requirements,
        )

        decision = RoutingDecision(
            entry_id=entry.id,
            queue_id=queue.id,
        )

        # Try primary strategy
        strategy = self._strategies.get(strategy_name)
        if strategy and available_agents:
            result = await strategy.select_agent(entry, available_agents)
            if result:
                agent, score, reason = result
                decision.agent_id = agent.id
                decision.routed = True
                decision.match_score = score
                decision.decision_reason = reason
                decision.strategy_used = queue.config.routing_strategy

                logger.info(
                    f"Routed entry {entry.id} to agent {agent.id} "
                    f"using {strategy_name}: {reason}"
                )

                self._decisions.append(decision)
                return decision

        # Try fallback strategy if different
        fallback_name = queue.config.fallback_strategy.value
        if fallback_name != strategy_name:
            fallback = self._strategies.get(fallback_name)
            if fallback and available_agents:
                result = await fallback.select_agent(entry, available_agents)
                if result:
                    agent, score, reason = result
                    decision.agent_id = agent.id
                    decision.routed = True
                    decision.match_score = score
                    decision.decision_reason = f"Fallback: {reason}"
                    decision.strategy_used = queue.config.fallback_strategy

                    self._decisions.append(decision)
                    return decision

        # No agent available - handle overflow
        decision = await self._handle_overflow(entry, queue, decision)
        self._decisions.append(decision)

        return decision

    async def _handle_overflow(
        self,
        entry: QueueEntry,
        queue: CallQueue,
        decision: RoutingDecision,
    ) -> RoutingDecision:
        """Handle overflow when no agents available."""
        overflow_action = queue.config.overflow_action

        decision.route_type = "overflow"
        decision.overflow_action = overflow_action
        decision.decision_reason = "No available agents"

        if overflow_action == OverflowAction.TRANSFER:
            # Transfer to another queue
            if queue.config.overflow_target_queue_id:
                decision.overflow_target = queue.config.overflow_target_queue_id
                decision.routed = True
                decision.decision_reason = f"Transferring to queue {queue.config.overflow_target_queue_id}"

        elif overflow_action == OverflowAction.CALLBACK:
            decision.route_type = "callback"
            decision.routed = True
            decision.decision_reason = "Callback scheduled"

        elif overflow_action == OverflowAction.VOICEMAIL:
            decision.route_type = "voicemail"
            decision.routed = True
            decision.decision_reason = "Routing to voicemail"

        elif overflow_action == OverflowAction.AI_AGENT:
            decision.route_type = "ai_agent"
            decision.routed = True
            decision.decision_reason = "Routing to AI agent"

        else:
            decision.decision_reason = f"Overflow action: {overflow_action.value}"

        logger.warning(
            f"Entry {entry.id} overflow: {decision.decision_reason}"
        )

        return decision

    async def get_routing_statistics(
        self,
        queue_id: Optional[str] = None,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """Get routing statistics."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        decisions = [
            d for d in self._decisions
            if d.decided_at >= cutoff
        ]

        if queue_id:
            decisions = [d for d in decisions if d.queue_id == queue_id]

        if not decisions:
            return {
                "total_decisions": 0,
                "routed": 0,
                "overflow": 0,
                "strategies": {},
            }

        routed = [d for d in decisions if d.routed and d.route_type == "agent"]
        overflow = [d for d in decisions if d.route_type == "overflow"]

        strategy_counts: Dict[str, int] = defaultdict(int)
        for d in routed:
            strategy_counts[d.strategy_used.value] += 1

        return {
            "total_decisions": len(decisions),
            "routed": len(routed),
            "overflow": len(overflow),
            "success_rate": len(routed) / len(decisions) if decisions else 0,
            "avg_match_score": sum(d.match_score for d in routed) / len(routed) if routed else 0,
            "strategies": dict(strategy_counts),
        }


# =============================================================================
# Callback Manager
# =============================================================================


class CallbackManager:
    """Manages callback requests and scheduling."""

    def __init__(self):
        self._callbacks: Dict[str, CallbackRequest] = {}
        self._callbacks_by_org: Dict[str, List[str]] = defaultdict(list)
        self._pending_callbacks: List[str] = []

    async def create_callback(
        self,
        organization_id: str,
        queue_id: str,
        entry_id: str,
        phone_number: str,
        caller_name: Optional[str] = None,
        preferred_time: Optional[datetime] = None,
        context: Optional[Dict[str, Any]] = None,
        max_attempts: int = 3,
    ) -> CallbackRequest:
        """Create a callback request."""
        callback = CallbackRequest(
            id=f"callback_{uuid.uuid4().hex[:18]}",
            organization_id=organization_id,
            queue_id=queue_id,
            original_entry_id=entry_id,
            phone_number=phone_number,
            caller_name=caller_name,
            preferred_time=preferred_time,
            max_attempts=max_attempts,
            context=context or {},
        )

        self._callbacks[callback.id] = callback
        self._callbacks_by_org[organization_id].append(callback.id)
        self._pending_callbacks.append(callback.id)

        logger.info(f"Created callback {callback.id} for {phone_number}")

        return callback

    async def get_callback(self, callback_id: str) -> Optional[CallbackRequest]:
        """Get callback by ID."""
        return self._callbacks.get(callback_id)

    async def get_pending_callbacks(
        self,
        organization_id: Optional[str] = None,
        queue_id: Optional[str] = None,
    ) -> List[CallbackRequest]:
        """Get pending callbacks."""
        callbacks = []

        for callback_id in self._pending_callbacks:
            callback = self._callbacks.get(callback_id)
            if not callback or not callback.can_attempt:
                continue

            if organization_id and callback.organization_id != organization_id:
                continue
            if queue_id and callback.queue_id != queue_id:
                continue

            callbacks.append(callback)

        # Sort by scheduled time
        return sorted(callbacks, key=lambda c: c.scheduled_at)

    async def get_next_callback(
        self,
        organization_id: str,
        queue_id: Optional[str] = None,
    ) -> Optional[CallbackRequest]:
        """Get the next callback to process."""
        callbacks = await self.get_pending_callbacks(organization_id, queue_id)

        for callback in callbacks:
            # Check if preferred time has passed
            if callback.preferred_time:
                if datetime.utcnow() < callback.preferred_time:
                    continue
            return callback

        return None

    async def record_attempt(
        self,
        callback_id: str,
        result: str,
        agent_id: Optional[str] = None,
    ) -> CallbackRequest:
        """Record a callback attempt."""
        callback = await self.get_callback(callback_id)
        if not callback:
            raise QueueRoutingError(f"Callback {callback_id} not found")

        callback.attempts += 1
        callback.last_attempt_at = datetime.utcnow()
        callback.last_attempt_result = result

        logger.info(
            f"Callback {callback_id} attempt {callback.attempts}: {result}"
        )

        return callback

    async def complete_callback(
        self,
        callback_id: str,
        agent_id: str,
    ) -> CallbackRequest:
        """Mark callback as completed."""
        callback = await self.get_callback(callback_id)
        if not callback:
            raise QueueRoutingError(f"Callback {callback_id} not found")

        callback.completed = True
        callback.completed_at = datetime.utcnow()
        callback.completed_by_agent_id = agent_id

        # Remove from pending
        if callback_id in self._pending_callbacks:
            self._pending_callbacks.remove(callback_id)

        logger.info(f"Callback {callback_id} completed by agent {agent_id}")

        return callback

    async def cancel_callback(
        self,
        callback_id: str,
        reason: str = "",
    ) -> CallbackRequest:
        """Cancel a callback request."""
        callback = await self.get_callback(callback_id)
        if not callback:
            raise QueueRoutingError(f"Callback {callback_id} not found")

        callback.completed = True
        callback.completed_at = datetime.utcnow()
        callback.notes = f"Cancelled: {reason}"

        if callback_id in self._pending_callbacks:
            self._pending_callbacks.remove(callback_id)

        logger.info(f"Callback {callback_id} cancelled: {reason}")

        return callback


# =============================================================================
# Analytics Service
# =============================================================================


class QueueAnalyticsService:
    """Provides queue and agent analytics."""

    def __init__(
        self,
        queue_manager: QueueManager,
        agent_manager: AgentManager,
    ):
        self._queue_manager = queue_manager
        self._agent_manager = agent_manager
        self._metrics_history: Dict[str, List[QueueMetrics]] = defaultdict(list)
        self._agent_metrics: Dict[str, AgentMetrics] = {}

    async def get_queue_metrics(self, queue_id: str) -> QueueMetrics:
        """Get current metrics for a queue."""
        queue = await self._queue_manager.get_queue(queue_id)
        if not queue:
            raise QueueNotFoundError(f"Queue {queue_id} not found")

        # Get agent counts
        agents = await self._agent_manager.list_agents(
            queue.organization_id,
            queue_id=queue_id,
        )
        available_agents = [a for a in agents if a.is_available]
        busy_agents = [a for a in agents if a.status == AgentStatus.BUSY]

        # Get waiting entries
        entries = await self._queue_manager.get_entries(queue_id, waiting_only=True)

        # Calculate wait times
        wait_times = [e.wait_seconds for e in entries]
        avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0
        max_wait = max(wait_times) if wait_times else 0
        min_wait = min(wait_times) if wait_times else 0

        metrics = QueueMetrics(
            queue_id=queue_id,
            current_waiting=len(entries),
            current_talking=sum(a.current_calls for a in agents),
            available_agents=len(available_agents),
            busy_agents=len(busy_agents),
            avg_wait_time_seconds=avg_wait,
            max_wait_time_seconds=max_wait,
            min_wait_time_seconds=min_wait,
            calls_offered=queue.total_calls_today,
            calls_answered=queue.calls_answered_today,
            calls_abandoned=queue.calls_abandoned_today,
            service_level_percent=queue.sla_current_percent,
            service_level_target_seconds=queue.config.target_answer_time_seconds,
            abandonment_rate_percent=queue.abandonment_rate,
        )

        # Store for history
        self._metrics_history[queue_id].append(metrics)

        return metrics

    async def get_agent_metrics(
        self,
        agent_id: str,
        period_hours: int = 24,
    ) -> AgentMetrics:
        """Get metrics for an agent."""
        agent = await self._agent_manager.get_agent(agent_id)
        if not agent:
            raise AgentNotFoundError(f"Agent {agent_id} not found")

        metrics = AgentMetrics(
            agent_id=agent_id,
            period_start=datetime.utcnow() - timedelta(hours=period_hours),
            calls_handled=agent.total_calls_handled,
            avg_handle_time_seconds=agent.avg_handle_time_seconds,
        )

        return metrics

    async def get_dashboard_data(
        self,
        organization_id: str,
    ) -> Dict[str, Any]:
        """Get dashboard data for organization."""
        queues = await self._queue_manager.list_queues(organization_id)
        agents = await self._agent_manager.list_agents(organization_id)

        queue_metrics = []
        for queue in queues:
            try:
                metrics = await self.get_queue_metrics(queue.id)
                queue_metrics.append(metrics.to_dict())
            except Exception as e:
                logger.error(f"Error getting metrics for queue {queue.id}: {e}")

        # Aggregate stats
        total_waiting = sum(m.get("current_waiting", 0) for m in queue_metrics)
        total_talking = sum(m.get("current_talking", 0) for m in queue_metrics)
        available_agents = [a for a in agents if a.is_available]
        busy_agents = [a for a in agents if a.status == AgentStatus.BUSY]

        return {
            "summary": {
                "total_queues": len(queues),
                "active_queues": sum(1 for q in queues if q.status == QueueStatus.ACTIVE),
                "total_agents": len(agents),
                "available_agents": len(available_agents),
                "busy_agents": len(busy_agents),
                "total_waiting": total_waiting,
                "total_talking": total_talking,
            },
            "queues": queue_metrics,
            "agents": [a.to_dict() for a in agents[:10]],  # Top 10
        }


# =============================================================================
# Main Service
# =============================================================================


class QueueRoutingService:
    """
    Unified service for queue management and call routing.

    Provides:
    - Queue management
    - Agent management
    - Skill management
    - Call routing
    - Callback handling
    - Analytics
    """

    def __init__(self):
        self.skills = SkillManager()
        self.agents = AgentManager()
        self.queues = QueueManager()
        self.routing = RoutingEngine(self.agents)
        self.callbacks = CallbackManager()
        self.analytics = QueueAnalyticsService(self.queues, self.agents)

    async def enqueue_call(
        self,
        queue_id: str,
        call_id: str,
        caller_phone: str,
        caller_name: Optional[str] = None,
        caller_id: Optional[str] = None,
        priority: QueuePriority = QueuePriority.NORMAL,
        is_vip: bool = False,
        context: Optional[Dict[str, Any]] = None,
        route_immediately: bool = True,
    ) -> Tuple[QueueEntry, Optional[RoutingDecision]]:
        """
        Add a call to the queue and optionally route immediately.

        Returns:
            Tuple of (entry, routing_decision)
        """
        # Add to queue
        entry = await self.queues.add_entry(
            queue_id=queue_id,
            call_id=call_id,
            caller_phone=caller_phone,
            caller_name=caller_name,
            caller_id=caller_id,
            priority=priority,
            is_vip=is_vip,
            context=context,
        )

        decision = None
        if route_immediately:
            queue = await self.queues.get_queue(queue_id)
            if queue:
                decision = await self.routing.route(entry, queue)

                # Handle successful routing
                if decision.routed and decision.agent_id:
                    await self.agents.assign_call(decision.agent_id)
                    await self.queues.complete_entry(
                        entry.id,
                        CallOutcome.ANSWERED,
                        decision.agent_id,
                    )

        return entry, decision

    async def process_queue(
        self,
        queue_id: str,
    ) -> List[RoutingDecision]:
        """
        Process all waiting entries in a queue.

        Returns:
            List of routing decisions made
        """
        queue = await self.queues.get_queue(queue_id)
        if not queue:
            raise QueueNotFoundError(f"Queue {queue_id} not found")

        decisions = []
        entries = await self.queues.get_entries(queue_id, waiting_only=True)

        for entry in entries:
            decision = await self.routing.route(entry, queue)
            decisions.append(decision)

            if decision.routed and decision.agent_id:
                await self.agents.assign_call(decision.agent_id)
                await self.queues.complete_entry(
                    entry.id,
                    CallOutcome.ANSWERED,
                    decision.agent_id,
                )

        return decisions

    async def complete_call(
        self,
        entry_id: str,
        agent_id: str,
        handle_time_seconds: float,
        wrap_up_time_seconds: float = 30.0,
    ) -> None:
        """Complete a call and release the agent."""
        await self.agents.release_call(
            agent_id,
            handle_time_seconds=handle_time_seconds,
            wrap_up_time_seconds=wrap_up_time_seconds,
        )

    async def abandon_call(self, entry_id: str) -> QueueEntry:
        """Mark a call as abandoned."""
        return await self.queues.complete_entry(
            entry_id,
            CallOutcome.ABANDONED,
        )

    async def transfer_call(
        self,
        entry_id: str,
        target_queue_id: str,
    ) -> Tuple[QueueEntry, Optional[RoutingDecision]]:
        """Transfer a call to another queue."""
        entry = await self.queues.get_entry(entry_id)
        if not entry:
            raise QueueRoutingError(f"Entry {entry_id} not found")

        # Mark original as transferred
        await self.queues.complete_entry(entry_id, CallOutcome.TRANSFERRED)

        # Create new entry in target queue
        new_entry = await self.queues.add_entry(
            queue_id=target_queue_id,
            call_id=entry.call_id,
            caller_phone=entry.caller_phone,
            caller_name=entry.caller_name,
            caller_id=entry.caller_id,
            priority=entry.priority,
            is_vip=entry.is_vip,
            context={
                **entry.context,
                "transfer_from_queue_id": entry.queue_id,
                "transfer_from_entry_id": entry_id,
            },
        )
        new_entry.source = "transfer"
        new_entry.transfer_from_queue_id = entry.queue_id

        # Route immediately
        target_queue = await self.queues.get_queue(target_queue_id)
        decision = None
        if target_queue:
            decision = await self.routing.route(new_entry, target_queue)

        return new_entry, decision

    async def schedule_callback(
        self,
        entry_id: str,
        phone_number: Optional[str] = None,
        preferred_time: Optional[datetime] = None,
    ) -> CallbackRequest:
        """Schedule a callback for a queue entry."""
        entry = await self.queues.get_entry(entry_id)
        if not entry:
            raise QueueRoutingError(f"Entry {entry_id} not found")

        # Create callback
        callback = await self.callbacks.create_callback(
            organization_id=entry.organization_id,
            queue_id=entry.queue_id,
            entry_id=entry_id,
            phone_number=phone_number or entry.caller_phone,
            caller_name=entry.caller_name,
            preferred_time=preferred_time,
            context=entry.context,
        )

        # Mark entry as callback scheduled
        await self.queues.complete_entry(entry_id, CallOutcome.CALLBACK_SCHEDULED)

        return callback

    async def get_queue_status(self, queue_id: str) -> Dict[str, Any]:
        """Get comprehensive status for a queue."""
        queue = await self.queues.get_queue(queue_id)
        if not queue:
            raise QueueNotFoundError(f"Queue {queue_id} not found")

        metrics = await self.analytics.get_queue_metrics(queue_id)
        entries = await self.queues.get_entries(queue_id, waiting_only=True)
        agents = await self.agents.list_agents(
            queue.organization_id,
            queue_id=queue_id,
        )

        return {
            "queue": queue.to_dict(),
            "metrics": metrics.to_dict(),
            "waiting_entries": [e.to_dict() for e in entries[:20]],
            "agents": [a.to_dict() for a in agents],
        }

    async def get_organization_dashboard(
        self,
        organization_id: str,
    ) -> Dict[str, Any]:
        """Get dashboard for entire organization."""
        return await self.analytics.get_dashboard_data(organization_id)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Managers
    "SkillManager",
    "AgentManager",
    "QueueManager",
    # Routing
    "RoutingStrategy",
    "RoundRobinStrategy",
    "LeastBusyStrategy",
    "LongestIdleStrategy",
    "SkillsBasedStrategy",
    "WeightedStrategy",
    "StickyStrategy",
    "RoutingEngine",
    # Callbacks
    "CallbackManager",
    # Analytics
    "QueueAnalyticsService",
    # Main Service
    "QueueRoutingService",
]
