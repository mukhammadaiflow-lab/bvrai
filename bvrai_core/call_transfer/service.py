"""
Call Transfer Service Module

This module provides the service layer for intelligent call transfer
and handoff management, including agent routing, context passing,
and transfer orchestration.

Key Services:
- AgentRegistry: Manage human and AI agents
- RoutingEngine: Skill-based and rule-based routing
- TransferOrchestrator: Execute and manage transfers
- ContextBuilder: Build transfer context
- TransferAnalytics: Track transfer metrics
- CallTransferService: Main orchestrating service
"""

import asyncio
import random
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from .base import (
    # Enums
    TransferType,
    TransferTargetType,
    TransferStatus,
    TransferReason,
    TransferPriority,
    RoutingStrategy,
    RoutingConditionType,
    # Target types
    HumanAgent,
    AgentGroup,
    Department,
    ExternalDestination,
    AIAgentTarget,
    TransferTarget,
    # Context types
    ConversationSummaryContext,
    TransferContext,
    # Request/Result types
    TransferRequest,
    TransferResult,
    # Routing types
    RoutingCondition,
    RoutingRule,
    # Analytics types
    TransferMetrics,
    AgentTransferMetrics,
    # Exceptions
    TransferError,
    TransferTargetNotFoundError,
    TransferTargetUnavailableError,
    TransferTimeoutError,
    TransferRejectedError,
    NoAvailableAgentsError,
    RoutingRuleNotFoundError,
)


# =============================================================================
# Agent Registry
# =============================================================================


class AgentRegistry:
    """
    Registry for managing agents and agent groups.

    Provides CRUD operations and availability tracking for
    human agents, AI agents, groups, and departments.
    """

    def __init__(self):
        # Human agents
        self._agents: Dict[str, HumanAgent] = {}
        self._agents_by_skill: Dict[str, Set[str]] = {}
        self._agents_by_department: Dict[str, Set[str]] = {}
        self._agents_by_language: Dict[str, Set[str]] = {}

        # Agent groups
        self._groups: Dict[str, AgentGroup] = {}

        # Departments
        self._departments: Dict[str, Department] = {}

        # AI agents
        self._ai_agents: Dict[str, AIAgentTarget] = {}

        # External destinations
        self._external: Dict[str, ExternalDestination] = {}

    # -------------------------------------------------------------------------
    # Human Agent Management
    # -------------------------------------------------------------------------

    def register_agent(self, agent: HumanAgent) -> HumanAgent:
        """Register a human agent."""
        self._agents[agent.id] = agent

        # Index by skills
        for skill in agent.skills:
            if skill not in self._agents_by_skill:
                self._agents_by_skill[skill] = set()
            self._agents_by_skill[skill].add(agent.id)

        # Index by department
        if agent.department:
            if agent.department not in self._agents_by_department:
                self._agents_by_department[agent.department] = set()
            self._agents_by_department[agent.department].add(agent.id)

        # Index by language
        for lang in agent.languages:
            if lang not in self._agents_by_language:
                self._agents_by_language[lang] = set()
            self._agents_by_language[lang].add(agent.id)

        return agent

    def get_agent(self, agent_id: str) -> Optional[HumanAgent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def get_all_agents(self) -> List[HumanAgent]:
        """Get all registered agents."""
        return list(self._agents.values())

    def get_available_agents(self) -> List[HumanAgent]:
        """Get all available agents."""
        return [a for a in self._agents.values() if a.can_accept_call]

    def get_agents_by_skill(self, skill: str) -> List[HumanAgent]:
        """Get agents with a specific skill."""
        agent_ids = self._agents_by_skill.get(skill, set())
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def get_agents_by_skills(
        self, skills: List[str], require_all: bool = False
    ) -> List[HumanAgent]:
        """Get agents matching skills."""
        if require_all:
            # Must have all skills
            candidates = set(self._agents.keys())
            for skill in skills:
                skill_agents = self._agents_by_skill.get(skill, set())
                candidates = candidates.intersection(skill_agents)
            return [self._agents[aid] for aid in candidates]
        else:
            # Any of the skills
            candidates = set()
            for skill in skills:
                candidates.update(self._agents_by_skill.get(skill, set()))
            return [self._agents[aid] for aid in candidates]

    def get_agents_by_department(self, department: str) -> List[HumanAgent]:
        """Get agents in a department."""
        agent_ids = self._agents_by_department.get(department, set())
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def get_agents_by_language(self, language: str) -> List[HumanAgent]:
        """Get agents who speak a language."""
        agent_ids = self._agents_by_language.get(language, set())
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def update_agent_status(
        self,
        agent_id: str,
        is_available: Optional[bool] = None,
        is_online: Optional[bool] = None,
        current_status: Optional[str] = None,
        current_calls: Optional[int] = None,
    ) -> Optional[HumanAgent]:
        """Update agent status."""
        agent = self._agents.get(agent_id)
        if not agent:
            return None

        if is_available is not None:
            agent.is_available = is_available
        if is_online is not None:
            agent.is_online = is_online
        if current_status is not None:
            agent.current_status = current_status
        if current_calls is not None:
            agent.current_calls = current_calls

        return agent

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent."""
        agent = self._agents.get(agent_id)
        if not agent:
            return False

        # Remove from indexes
        for skill in agent.skills:
            if skill in self._agents_by_skill:
                self._agents_by_skill[skill].discard(agent_id)

        if agent.department in self._agents_by_department:
            self._agents_by_department[agent.department].discard(agent_id)

        for lang in agent.languages:
            if lang in self._agents_by_language:
                self._agents_by_language[lang].discard(agent_id)

        del self._agents[agent_id]
        return True

    # -------------------------------------------------------------------------
    # Agent Group Management
    # -------------------------------------------------------------------------

    def register_group(self, group: AgentGroup) -> AgentGroup:
        """Register an agent group."""
        self._groups[group.id] = group
        return group

    def get_group(self, group_id: str) -> Optional[AgentGroup]:
        """Get a group by ID."""
        return self._groups.get(group_id)

    def get_all_groups(self) -> List[AgentGroup]:
        """Get all groups."""
        return list(self._groups.values())

    def get_group_agents(self, group_id: str) -> List[HumanAgent]:
        """Get all agents in a group."""
        group = self._groups.get(group_id)
        if not group:
            return []
        return [self._agents[aid] for aid in group.agent_ids if aid in self._agents]

    def get_available_group_agents(self, group_id: str) -> List[HumanAgent]:
        """Get available agents in a group."""
        agents = self.get_group_agents(group_id)
        return [a for a in agents if a.can_accept_call]

    # -------------------------------------------------------------------------
    # Department Management
    # -------------------------------------------------------------------------

    def register_department(self, department: Department) -> Department:
        """Register a department."""
        self._departments[department.id] = department
        return department

    def get_department(self, department_id: str) -> Optional[Department]:
        """Get a department by ID."""
        return self._departments.get(department_id)

    def get_all_departments(self) -> List[Department]:
        """Get all departments."""
        return list(self._departments.values())

    # -------------------------------------------------------------------------
    # AI Agent Management
    # -------------------------------------------------------------------------

    def register_ai_agent(self, ai_agent: AIAgentTarget) -> AIAgentTarget:
        """Register an AI agent target."""
        self._ai_agents[ai_agent.id] = ai_agent
        return ai_agent

    def get_ai_agent(self, ai_agent_id: str) -> Optional[AIAgentTarget]:
        """Get an AI agent by ID."""
        return self._ai_agents.get(ai_agent_id)

    def get_available_ai_agents(self) -> List[AIAgentTarget]:
        """Get available AI agents."""
        return [a for a in self._ai_agents.values() if a.can_accept_call]

    # -------------------------------------------------------------------------
    # External Destination Management
    # -------------------------------------------------------------------------

    def register_external(self, external: ExternalDestination) -> ExternalDestination:
        """Register an external destination."""
        self._external[external.id] = external
        return external

    def get_external(self, external_id: str) -> Optional[ExternalDestination]:
        """Get an external destination by ID."""
        return self._external.get(external_id)

    def get_all_external(self) -> List[ExternalDestination]:
        """Get all external destinations."""
        return list(self._external.values())


# =============================================================================
# Routing Engine
# =============================================================================


class RoutingEngine:
    """
    Routing engine for intelligent agent selection.

    Implements various routing strategies including skill-based,
    round-robin, least-busy, and rule-based routing.
    """

    def __init__(self, registry: AgentRegistry):
        self._registry = registry
        self._rules: Dict[str, RoutingRule] = {}
        self._round_robin_index: Dict[str, int] = {}

    # -------------------------------------------------------------------------
    # Rule Management
    # -------------------------------------------------------------------------

    def add_rule(self, rule: RoutingRule) -> RoutingRule:
        """Add a routing rule."""
        self._rules[rule.id] = rule
        return rule

    def get_rule(self, rule_id: str) -> Optional[RoutingRule]:
        """Get a rule by ID."""
        return self._rules.get(rule_id)

    def get_all_rules(self, organization_id: str) -> List[RoutingRule]:
        """Get all rules for an organization."""
        return [
            r for r in self._rules.values()
            if r.organization_id == organization_id and r.is_active
        ]

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a routing rule."""
        if rule_id in self._rules:
            del self._rules[rule_id]
            return True
        return False

    # -------------------------------------------------------------------------
    # Routing Methods
    # -------------------------------------------------------------------------

    def find_best_target(
        self,
        request: TransferRequest,
    ) -> Optional[TransferTarget]:
        """
        Find the best transfer target for a request.

        Uses rules and strategies to select the optimal target.
        """
        # Try rule-based routing first
        matching_rules = self._get_matching_rules(request)

        for rule in matching_rules:
            target = self._route_by_rule(request, rule)
            if target:
                return target

        # Fall back to skill-based routing
        if request.target_skills:
            target = self._route_by_skills(request)
            if target:
                return target

        # Fall back to department routing
        if request.target_department:
            target = self._route_to_department(request)
            if target:
                return target

        # Fall back to agent group routing
        if request.target_agent_group:
            target = self._route_to_group(request)
            if target:
                return target

        return None

    def _get_matching_rules(
        self, request: TransferRequest
    ) -> List[RoutingRule]:
        """Get rules that match the request."""
        matching = []

        for rule in self._rules.values():
            if not rule.is_active:
                continue
            if rule.organization_id != request.organization_id:
                continue

            # Check all conditions
            if self._check_conditions(rule.conditions, request):
                matching.append(rule)

        # Sort by priority (lower = higher priority)
        matching.sort(key=lambda r: r.priority)

        return matching

    def _check_conditions(
        self,
        conditions: List[RoutingCondition],
        request: TransferRequest,
    ) -> bool:
        """Check if all conditions match."""
        for condition in conditions:
            if not self._check_condition(condition, request):
                return False
        return True

    def _check_condition(
        self,
        condition: RoutingCondition,
        request: TransferRequest,
    ) -> bool:
        """Check a single condition."""
        if condition.condition_type == RoutingConditionType.SKILL_MATCH:
            return condition.value in request.target_skills

        if condition.condition_type == RoutingConditionType.LANGUAGE_MATCH:
            return condition.value == request.context.preferred_language

        if condition.condition_type == RoutingConditionType.DEPARTMENT_MATCH:
            return condition.value == request.target_department

        if condition.condition_type == RoutingConditionType.CUSTOMER_TYPE:
            return request.context.is_vip if condition.value == "vip" else True

        if condition.condition_type == RoutingConditionType.SENTIMENT:
            sentiment = request.context.conversation_summary.customer_sentiment
            if condition.operator == "equals":
                return sentiment == condition.value
            if condition.operator == "less_than":
                # negative < neutral < positive
                order = {"negative": 0, "neutral": 1, "positive": 2}
                return order.get(sentiment, 1) < order.get(condition.value, 1)

        return True

    def _route_by_rule(
        self,
        request: TransferRequest,
        rule: RoutingRule,
    ) -> Optional[TransferTarget]:
        """Route using a specific rule."""
        # Get candidate targets based on rule target type
        if rule.target_type == TransferTargetType.HUMAN_AGENT:
            candidates = [
                self._registry.get_agent(tid)
                for tid in rule.target_ids
                if self._registry.get_agent(tid)
            ]
            candidates = [c for c in candidates if c and c.can_accept_call]
        elif rule.target_type == TransferTargetType.AGENT_GROUP:
            candidates = []
            for group_id in rule.target_ids:
                candidates.extend(
                    self._registry.get_available_group_agents(group_id)
                )
        elif rule.target_type == TransferTargetType.DEPARTMENT:
            candidates = []
            for dept_id in rule.target_ids:
                dept = self._registry.get_department(dept_id)
                if dept:
                    for group_id in dept.agent_group_ids:
                        candidates.extend(
                            self._registry.get_available_group_agents(group_id)
                        )
        elif rule.target_type == TransferTargetType.AI_AGENT:
            candidates = [
                self._registry.get_ai_agent(tid)
                for tid in rule.target_ids
                if self._registry.get_ai_agent(tid)
            ]
            candidates = [c for c in candidates if c and c.can_accept_call]
        else:
            candidates = []

        if not candidates:
            return None

        # Select based on strategy
        selected = self._select_by_strategy(rule.strategy, candidates, rule.id)

        if not selected:
            return None

        # Build transfer target
        return self._build_target(selected, rule.target_type)

    def _route_by_skills(
        self, request: TransferRequest
    ) -> Optional[TransferTarget]:
        """Route based on required skills."""
        candidates = self._registry.get_agents_by_skills(request.target_skills)
        available = [c for c in candidates if c.can_accept_call]

        if not available:
            return None

        # Score by skill match
        scored = []
        for agent in available:
            score = sum(
                1 for skill in request.target_skills
                if skill in agent.skills
            )
            scored.append((agent, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        selected = scored[0][0]
        return self._build_target(selected, TransferTargetType.HUMAN_AGENT)

    def _route_to_department(
        self, request: TransferRequest
    ) -> Optional[TransferTarget]:
        """Route to a department."""
        dept = self._registry.get_department(request.target_department)
        if not dept:
            return None

        # Get available agents in department
        candidates = []
        for group_id in dept.agent_group_ids:
            candidates.extend(self._registry.get_available_group_agents(group_id))

        if not candidates:
            # Return department target for queue
            return TransferTarget(
                id="",
                target_type=TransferTargetType.DEPARTMENT,
                name=dept.name,
                target_id=dept.id,
                phone_number=dept.main_number,
                sip_uri=dept.sip_uri,
            )

        # Select agent
        selected = self._select_by_strategy(
            RoutingStrategy.LEAST_BUSY, candidates, dept.id
        )

        if selected:
            return self._build_target(selected, TransferTargetType.HUMAN_AGENT)

        return None

    def _route_to_group(
        self, request: TransferRequest
    ) -> Optional[TransferTarget]:
        """Route to an agent group."""
        group = self._registry.get_group(request.target_agent_group)
        if not group:
            return None

        candidates = self._registry.get_available_group_agents(group.id)

        if not candidates:
            # Return group target for queue
            return TransferTarget(
                id="",
                target_type=TransferTargetType.AGENT_GROUP,
                name=group.name,
                target_id=group.id,
            )

        # Select based on group's routing strategy
        strategy = RoutingStrategy(group.routing_strategy)
        selected = self._select_by_strategy(strategy, candidates, group.id)

        if selected:
            return self._build_target(selected, TransferTargetType.HUMAN_AGENT)

        return None

    def _select_by_strategy(
        self,
        strategy: RoutingStrategy,
        candidates: List[Any],
        context_id: str,
    ) -> Optional[Any]:
        """Select a candidate using the specified strategy."""
        if not candidates:
            return None

        if strategy == RoutingStrategy.ROUND_ROBIN:
            index = self._round_robin_index.get(context_id, 0)
            selected = candidates[index % len(candidates)]
            self._round_robin_index[context_id] = index + 1
            return selected

        elif strategy == RoutingStrategy.LEAST_BUSY:
            # Sort by current calls (ascending)
            sorted_candidates = sorted(
                candidates,
                key=lambda c: getattr(c, 'current_calls', 0)
            )
            return sorted_candidates[0]

        elif strategy == RoutingStrategy.LONGEST_IDLE:
            # For now, random selection (would need idle time tracking)
            return random.choice(candidates)

        elif strategy == RoutingStrategy.WEIGHTED_RANDOM:
            return random.choice(candidates)

        elif strategy == RoutingStrategy.DIRECT:
            return candidates[0] if candidates else None

        else:
            # Default to first available
            return candidates[0]

    def _build_target(
        self,
        selected: Any,
        target_type: TransferTargetType,
    ) -> TransferTarget:
        """Build a TransferTarget from a selected candidate."""
        if isinstance(selected, HumanAgent):
            return TransferTarget(
                id="",
                target_type=TransferTargetType.HUMAN_AGENT,
                name=selected.name,
                target_id=selected.id,
                extension=selected.extension,
                sip_uri=selected.sip_uri,
            )
        elif isinstance(selected, AIAgentTarget):
            return TransferTarget(
                id="",
                target_type=TransferTargetType.AI_AGENT,
                name=selected.name,
                target_id=selected.agent_id,
            )
        else:
            return TransferTarget(
                id="",
                target_type=target_type,
                name=str(selected),
            )


# =============================================================================
# Transfer Orchestrator
# =============================================================================


class TransferOrchestrator:
    """
    Orchestrates the transfer process.

    Handles the actual execution of transfers including
    warm handoffs, cold transfers, and conference calls.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        routing_engine: RoutingEngine,
    ):
        self._registry = registry
        self._routing = routing_engine
        self._active_transfers: Dict[str, TransferRequest] = {}
        self._transfer_history: List[TransferResult] = []

    async def initiate_transfer(
        self, request: TransferRequest
    ) -> TransferResult:
        """
        Initiate a transfer.

        This is the main entry point for starting a transfer.
        """
        request.status = TransferStatus.INITIATING
        request.updated_at = datetime.utcnow()
        self._active_transfers[request.id] = request

        try:
            # Find target if not specified
            if not request.target:
                request.target = self._routing.find_best_target(request)

            if not request.target:
                raise NoAvailableAgentsError(
                    "No available agents for transfer"
                )

            # Execute transfer based on type
            if request.transfer_type == TransferType.WARM:
                result = await self._execute_warm_transfer(request)
            elif request.transfer_type == TransferType.COLD:
                result = await self._execute_cold_transfer(request)
            elif request.transfer_type == TransferType.BLIND:
                result = await self._execute_blind_transfer(request)
            elif request.transfer_type == TransferType.CONSULTATIVE:
                result = await self._execute_consultative_transfer(request)
            elif request.transfer_type == TransferType.CONFERENCE:
                result = await self._execute_conference(request)
            else:
                result = await self._execute_cold_transfer(request)

            # Record result
            self._transfer_history.append(result)

            return result

        except Exception as e:
            result = TransferResult(
                request_id=request.id,
                success=False,
                status=TransferStatus.FAILED,
                error_message=str(e),
            )
            request.status = TransferStatus.FAILED
            self._transfer_history.append(result)
            return result

        finally:
            # Remove from active transfers
            self._active_transfers.pop(request.id, None)

    async def _execute_warm_transfer(
        self, request: TransferRequest
    ) -> TransferResult:
        """Execute a warm transfer."""
        start_time = datetime.utcnow()
        request.status = TransferStatus.RINGING

        # Simulate ringing (in real implementation, this would call telephony API)
        ring_time = await self._ring_target(request.target, request.timeout_seconds)

        if ring_time < 0:
            # No answer
            return TransferResult(
                request_id=request.id,
                success=False,
                status=TransferStatus.NO_ANSWER,
                error_message="Target did not answer",
            )

        request.status = TransferStatus.CONNECTED

        # Simulate warm handoff (announce caller, pass context)
        await self._announce_transfer(request)

        request.status = TransferStatus.IN_PROGRESS

        # Wait for acceptance
        accepted = await self._wait_for_acceptance(request)

        if not accepted:
            return TransferResult(
                request_id=request.id,
                success=False,
                status=TransferStatus.REJECTED,
                error_message="Transfer rejected by target",
            )

        # Complete transfer
        request.status = TransferStatus.COMPLETED
        end_time = datetime.utcnow()

        # Update agent status
        if request.target.target_type == TransferTargetType.HUMAN_AGENT:
            self._registry.update_agent_status(
                request.target.target_id,
                current_calls=1,  # Increment would be better
            )

        return TransferResult(
            request_id=request.id,
            success=True,
            status=TransferStatus.COMPLETED,
            connected_to=request.target.target_id,
            connected_name=request.target.name,
            target_type=request.target.target_type,
            ring_time_seconds=ring_time,
            total_time_seconds=(end_time - start_time).total_seconds(),
        )

    async def _execute_cold_transfer(
        self, request: TransferRequest
    ) -> TransferResult:
        """Execute a cold transfer."""
        start_time = datetime.utcnow()
        request.status = TransferStatus.RINGING

        # Ring target
        ring_time = await self._ring_target(request.target, request.timeout_seconds)

        if ring_time < 0:
            return TransferResult(
                request_id=request.id,
                success=False,
                status=TransferStatus.NO_ANSWER,
            )

        # Immediate transfer (no announcement in cold transfer)
        request.status = TransferStatus.COMPLETED
        end_time = datetime.utcnow()

        return TransferResult(
            request_id=request.id,
            success=True,
            status=TransferStatus.COMPLETED,
            connected_to=request.target.target_id,
            connected_name=request.target.name,
            target_type=request.target.target_type,
            ring_time_seconds=ring_time,
            total_time_seconds=(end_time - start_time).total_seconds(),
        )

    async def _execute_blind_transfer(
        self, request: TransferRequest
    ) -> TransferResult:
        """Execute a blind transfer."""
        # Similar to cold transfer but with no verification
        request.status = TransferStatus.INITIATING

        # Just initiate transfer without waiting
        request.status = TransferStatus.COMPLETED

        return TransferResult(
            request_id=request.id,
            success=True,
            status=TransferStatus.COMPLETED,
            connected_to=request.target.target_id,
            connected_name=request.target.name,
            target_type=request.target.target_type,
        )

    async def _execute_consultative_transfer(
        self, request: TransferRequest
    ) -> TransferResult:
        """Execute a consultative transfer."""
        start_time = datetime.utcnow()

        # First, consult with target (while customer is on hold)
        request.status = TransferStatus.RINGING
        ring_time = await self._ring_target(request.target, request.timeout_seconds)

        if ring_time < 0:
            return TransferResult(
                request_id=request.id,
                success=False,
                status=TransferStatus.NO_ANSWER,
            )

        request.status = TransferStatus.CONNECTED

        # Consult with target agent
        consultation_result = await self._consult_with_target(request)

        if not consultation_result:
            return TransferResult(
                request_id=request.id,
                success=False,
                status=TransferStatus.REJECTED,
                error_message="Target declined after consultation",
            )

        # Now complete the transfer
        request.status = TransferStatus.COMPLETED
        end_time = datetime.utcnow()

        return TransferResult(
            request_id=request.id,
            success=True,
            status=TransferStatus.COMPLETED,
            connected_to=request.target.target_id,
            connected_name=request.target.name,
            target_type=request.target.target_type,
            ring_time_seconds=ring_time,
            total_time_seconds=(end_time - start_time).total_seconds(),
        )

    async def _execute_conference(
        self, request: TransferRequest
    ) -> TransferResult:
        """Execute a conference (all parties stay connected)."""
        request.status = TransferStatus.RINGING

        ring_time = await self._ring_target(request.target, request.timeout_seconds)

        if ring_time < 0:
            return TransferResult(
                request_id=request.id,
                success=False,
                status=TransferStatus.NO_ANSWER,
            )

        request.status = TransferStatus.CONNECTED

        # In conference mode, all parties are connected
        return TransferResult(
            request_id=request.id,
            success=True,
            status=TransferStatus.CONNECTED,  # Stays connected
            connected_to=request.target.target_id,
            connected_name=request.target.name,
            target_type=request.target.target_type,
            ring_time_seconds=ring_time,
        )

    # -------------------------------------------------------------------------
    # Helper Methods (Simulated - would integrate with telephony in production)
    # -------------------------------------------------------------------------

    async def _ring_target(
        self, target: TransferTarget, timeout_seconds: int
    ) -> float:
        """
        Ring the target and wait for answer.

        Returns ring time in seconds, or -1 if no answer.
        """
        # Simulate ringing (in production, this would use telephony API)
        await asyncio.sleep(0.1)  # Simulated delay

        # Simulate 90% answer rate
        if random.random() > 0.1:
            return random.uniform(2.0, 8.0)  # Simulated ring time
        return -1

    async def _announce_transfer(self, request: TransferRequest) -> None:
        """Announce the transfer to the target agent."""
        # In production, this would play an announcement or send context
        # to the target agent's interface
        await asyncio.sleep(0.05)

    async def _wait_for_acceptance(self, request: TransferRequest) -> bool:
        """Wait for target to accept the transfer."""
        # In production, this would wait for actual acceptance
        await asyncio.sleep(0.05)
        return random.random() > 0.05  # 95% acceptance rate

    async def _consult_with_target(self, request: TransferRequest) -> bool:
        """Consult with target before completing transfer."""
        # In production, this would enable two-way communication
        # between source agent and target before transfer
        await asyncio.sleep(0.05)
        return random.random() > 0.1  # 90% proceed after consultation

    # -------------------------------------------------------------------------
    # Transfer Management
    # -------------------------------------------------------------------------

    def get_active_transfers(self) -> List[TransferRequest]:
        """Get all active transfers."""
        return list(self._active_transfers.values())

    def get_transfer_status(
        self, transfer_id: str
    ) -> Optional[TransferRequest]:
        """Get status of a transfer."""
        return self._active_transfers.get(transfer_id)

    async def cancel_transfer(self, transfer_id: str) -> bool:
        """Cancel an active transfer."""
        request = self._active_transfers.get(transfer_id)
        if not request:
            return False

        request.status = TransferStatus.CANCELLED
        request.updated_at = datetime.utcnow()

        result = TransferResult(
            request_id=transfer_id,
            success=False,
            status=TransferStatus.CANCELLED,
        )
        self._transfer_history.append(result)

        return True

    def get_transfer_history(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TransferResult]:
        """Get transfer history."""
        return self._transfer_history[offset:offset + limit]


# =============================================================================
# Context Builder
# =============================================================================


class ContextBuilder:
    """
    Builds transfer context from conversation data.

    Extracts relevant information to pass during handoffs.
    """

    def build_context(
        self,
        call_id: str,
        conversation_data: Dict[str, Any],
        customer_data: Optional[Dict[str, Any]] = None,
        analysis_data: Optional[Dict[str, Any]] = None,
    ) -> TransferContext:
        """Build transfer context from various data sources."""
        context = TransferContext(call_id=call_id)

        # Build conversation summary
        summary = ConversationSummaryContext()

        if conversation_data:
            summary.summary = conversation_data.get("summary", "")
            summary.issue_type = conversation_data.get("issue_type", "")
            summary.issue_description = conversation_data.get("description", "")
            summary.pending_actions = conversation_data.get("pending_actions", [])
            summary.agent_notes = conversation_data.get("notes", "")

        if customer_data:
            summary.customer_name = customer_data.get("name", "")
            summary.customer_id = customer_data.get("id", "")
            summary.customer_phone = customer_data.get("phone", "")
            summary.customer_email = customer_data.get("email", "")
            context.is_vip = customer_data.get("is_vip", False)

        if analysis_data:
            summary.customer_sentiment = analysis_data.get("sentiment", "neutral")
            summary.sentiment_score = analysis_data.get("sentiment_score", 0.0)
            summary.entities = analysis_data.get("entities", {})

            # Set priority based on sentiment
            if summary.sentiment_score < -0.5:
                context.priority = TransferPriority.HIGH

        context.conversation_summary = summary

        return context

    def extract_skills_needed(
        self,
        issue_type: str,
        entities: Dict[str, Any],
    ) -> List[str]:
        """Extract required skills from issue data."""
        skills = []

        # Map issue types to skills
        skill_mapping = {
            "technical": ["technical_support", "troubleshooting"],
            "billing": ["billing", "payments"],
            "sales": ["sales", "upselling"],
            "account": ["account_management"],
            "returns": ["returns", "refunds"],
        }

        for key, skill_list in skill_mapping.items():
            if key in issue_type.lower():
                skills.extend(skill_list)

        # Extract from entities
        if "product" in entities:
            skills.append("product_specialist")

        return list(set(skills))


# =============================================================================
# Transfer Analytics
# =============================================================================


class TransferAnalytics:
    """
    Analytics for transfer operations.

    Tracks metrics, generates reports, and identifies patterns.
    """

    def __init__(self):
        self._metrics: Dict[str, TransferMetrics] = {}
        self._agent_metrics: Dict[str, AgentTransferMetrics] = {}

    def record_transfer(
        self,
        request: TransferRequest,
        result: TransferResult,
    ) -> None:
        """Record a transfer for analytics."""
        org_id = request.organization_id

        # Get or create org metrics
        if org_id not in self._metrics:
            self._metrics[org_id] = TransferMetrics()

        metrics = self._metrics[org_id]

        # Update counts
        metrics.total_transfers += 1
        if result.success:
            metrics.successful_transfers += 1
        else:
            metrics.failed_transfers += 1

        # Update by type
        if request.transfer_type not in metrics.transfers_by_type:
            metrics.transfers_by_type[request.transfer_type] = 0
        metrics.transfers_by_type[request.transfer_type] += 1

        # Update by reason
        if request.reason not in metrics.transfers_by_reason:
            metrics.transfers_by_reason[request.reason] = 0
        metrics.transfers_by_reason[request.reason] += 1

        # Update timing
        if result.ring_time_seconds > 0:
            # Update running average
            n = metrics.total_transfers
            metrics.average_ring_time_seconds = (
                (metrics.average_ring_time_seconds * (n - 1) + result.ring_time_seconds) / n
            )

        # Calculate success rate
        metrics.success_rate = metrics.calculate_success_rate()

        # Update agent metrics
        if request.source_agent_id:
            self._record_agent_transfer_out(request)

        if result.connected_to:
            self._record_agent_transfer_in(result)

    def _record_agent_transfer_out(self, request: TransferRequest) -> None:
        """Record an outgoing transfer for an agent."""
        agent_id = request.source_agent_id

        if agent_id not in self._agent_metrics:
            self._agent_metrics[agent_id] = AgentTransferMetrics(
                agent_id=agent_id,
                agent_name="",  # Would be populated from registry
            )

        metrics = self._agent_metrics[agent_id]
        metrics.transfers_out += 1

        if request.reason not in metrics.transfer_out_reasons:
            metrics.transfer_out_reasons[request.reason] = 0
        metrics.transfer_out_reasons[request.reason] += 1

    def _record_agent_transfer_in(self, result: TransferResult) -> None:
        """Record an incoming transfer for an agent."""
        agent_id = result.connected_to

        if agent_id not in self._agent_metrics:
            self._agent_metrics[agent_id] = AgentTransferMetrics(
                agent_id=agent_id,
                agent_name=result.connected_name,
            )

        metrics = self._agent_metrics[agent_id]
        metrics.transfers_in += 1

        if result.success:
            metrics.transfers_accepted += 1
        elif result.status == TransferStatus.REJECTED:
            metrics.transfers_rejected += 1
        elif result.status == TransferStatus.NO_ANSWER:
            metrics.transfers_missed += 1

    def get_metrics(
        self,
        organization_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> TransferMetrics:
        """Get transfer metrics for an organization."""
        return self._metrics.get(organization_id, TransferMetrics())

    def get_agent_metrics(self, agent_id: str) -> AgentTransferMetrics:
        """Get transfer metrics for an agent."""
        return self._agent_metrics.get(
            agent_id,
            AgentTransferMetrics(agent_id=agent_id, agent_name="")
        )

    def get_top_transfer_reasons(
        self,
        organization_id: str,
        limit: int = 5,
    ) -> List[Tuple[TransferReason, int]]:
        """Get top transfer reasons."""
        metrics = self._metrics.get(organization_id)
        if not metrics:
            return []

        sorted_reasons = sorted(
            metrics.transfers_by_reason.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_reasons[:limit]


# =============================================================================
# Main Call Transfer Service
# =============================================================================


class CallTransferService:
    """
    Main service for call transfer management.

    This is the primary entry point for all transfer operations.
    """

    def __init__(self):
        # Initialize components
        self.registry = AgentRegistry()
        self.routing = RoutingEngine(self.registry)
        self.orchestrator = TransferOrchestrator(self.registry, self.routing)
        self.context_builder = ContextBuilder()
        self.analytics = TransferAnalytics()

    # -------------------------------------------------------------------------
    # Agent Management
    # -------------------------------------------------------------------------

    def register_human_agent(
        self,
        name: str,
        email: str,
        skills: List[str] = None,
        languages: List[str] = None,
        department: str = "",
        extension: str = "",
        **kwargs,
    ) -> HumanAgent:
        """Register a human agent."""
        agent = HumanAgent(
            id="",
            name=name,
            email=email,
            skills=skills or [],
            languages=languages or ["en"],
            department=department,
            extension=extension,
            **kwargs,
        )
        return self.registry.register_agent(agent)

    def register_agent_group(
        self,
        name: str,
        agent_ids: List[str] = None,
        skills: List[str] = None,
        routing_strategy: str = "round_robin",
        **kwargs,
    ) -> AgentGroup:
        """Register an agent group."""
        group = AgentGroup(
            id="",
            name=name,
            agent_ids=agent_ids or [],
            skills=skills or [],
            routing_strategy=routing_strategy,
            **kwargs,
        )
        return self.registry.register_group(group)

    def register_department(
        self,
        name: str,
        agent_group_ids: List[str] = None,
        main_number: str = "",
        **kwargs,
    ) -> Department:
        """Register a department."""
        dept = Department(
            id="",
            name=name,
            agent_group_ids=agent_group_ids or [],
            main_number=main_number,
            **kwargs,
        )
        return self.registry.register_department(dept)

    def register_ai_agent_target(
        self,
        name: str,
        agent_id: str,
        skills: List[str] = None,
        languages: List[str] = None,
    ) -> AIAgentTarget:
        """Register an AI agent as transfer target."""
        target = AIAgentTarget(
            id="",
            name=name,
            agent_id=agent_id,
            skills=skills or [],
            languages=languages or ["en"],
        )
        return self.registry.register_ai_agent(target)

    def update_agent_availability(
        self,
        agent_id: str,
        is_available: bool,
        current_status: str = None,
    ) -> Optional[HumanAgent]:
        """Update agent availability."""
        return self.registry.update_agent_status(
            agent_id,
            is_available=is_available,
            current_status=current_status,
        )

    def get_available_agents(
        self,
        skills: List[str] = None,
        department: str = None,
        language: str = None,
    ) -> List[HumanAgent]:
        """Get available agents matching criteria."""
        agents = self.registry.get_available_agents()

        if skills:
            agents = [
                a for a in agents
                if any(s in a.skills for s in skills)
            ]

        if department:
            agents = [a for a in agents if a.department == department]

        if language:
            agents = [a for a in agents if language in a.languages]

        return agents

    # -------------------------------------------------------------------------
    # Routing Rules
    # -------------------------------------------------------------------------

    def add_routing_rule(
        self,
        organization_id: str,
        name: str,
        conditions: List[Dict[str, Any]],
        target_type: TransferTargetType,
        target_ids: List[str],
        strategy: RoutingStrategy = RoutingStrategy.SKILL_BASED,
        priority: int = 100,
    ) -> RoutingRule:
        """Add a routing rule."""
        rule_conditions = [
            RoutingCondition(
                condition_type=RoutingConditionType(c.get("type")),
                operator=c.get("operator", "equals"),
                value=c.get("value"),
                weight=c.get("weight", 1.0),
            )
            for c in conditions
        ]

        rule = RoutingRule(
            id="",
            name=name,
            organization_id=organization_id,
            priority=priority,
            conditions=rule_conditions,
            strategy=strategy,
            target_type=target_type,
            target_ids=target_ids,
        )

        return self.routing.add_rule(rule)

    def get_routing_rules(self, organization_id: str) -> List[RoutingRule]:
        """Get routing rules for an organization."""
        return self.routing.get_all_rules(organization_id)

    # -------------------------------------------------------------------------
    # Transfer Operations
    # -------------------------------------------------------------------------

    async def initiate_transfer(
        self,
        organization_id: str,
        call_id: str,
        transfer_type: TransferType = TransferType.WARM,
        reason: TransferReason = TransferReason.OTHER,
        target_skills: List[str] = None,
        target_department: str = None,
        target_agent_id: str = None,
        context: TransferContext = None,
        priority: TransferPriority = TransferPriority.NORMAL,
        **kwargs,
    ) -> TransferResult:
        """
        Initiate a call transfer.

        This is the main method for starting transfers.
        """
        # Build request
        request = TransferRequest(
            id="",
            organization_id=organization_id,
            call_id=call_id,
            transfer_type=transfer_type,
            reason=reason,
            priority=priority,
            target_skills=target_skills or [],
            target_department=target_department or "",
            context=context or TransferContext(),
        )

        # Set direct target if specified
        if target_agent_id:
            agent = self.registry.get_agent(target_agent_id)
            if agent:
                request.target = TransferTarget(
                    id="",
                    target_type=TransferTargetType.HUMAN_AGENT,
                    name=agent.name,
                    target_id=agent.id,
                    extension=agent.extension,
                    sip_uri=agent.sip_uri,
                )

        # Execute transfer
        result = await self.orchestrator.initiate_transfer(request)

        # Record analytics
        self.analytics.record_transfer(request, result)

        return result

    async def transfer_to_department(
        self,
        organization_id: str,
        call_id: str,
        department_id: str,
        transfer_type: TransferType = TransferType.WARM,
        context: TransferContext = None,
    ) -> TransferResult:
        """Transfer to a department."""
        return await self.initiate_transfer(
            organization_id=organization_id,
            call_id=call_id,
            transfer_type=transfer_type,
            target_department=department_id,
            context=context,
        )

    async def transfer_to_skills(
        self,
        organization_id: str,
        call_id: str,
        skills: List[str],
        transfer_type: TransferType = TransferType.WARM,
        context: TransferContext = None,
    ) -> TransferResult:
        """Transfer to an agent with specific skills."""
        return await self.initiate_transfer(
            organization_id=organization_id,
            call_id=call_id,
            transfer_type=transfer_type,
            target_skills=skills,
            context=context,
        )

    async def escalate_call(
        self,
        organization_id: str,
        call_id: str,
        reason: TransferReason = TransferReason.SUPERVISOR_NEEDED,
        context: TransferContext = None,
    ) -> TransferResult:
        """Escalate a call to supervisor."""
        return await self.initiate_transfer(
            organization_id=organization_id,
            call_id=call_id,
            transfer_type=TransferType.ESCALATION,
            reason=reason,
            target_skills=["supervisor"],
            priority=TransferPriority.HIGH,
            context=context,
        )

    async def cancel_transfer(self, transfer_id: str) -> bool:
        """Cancel an active transfer."""
        return await self.orchestrator.cancel_transfer(transfer_id)

    def get_transfer_status(
        self, transfer_id: str
    ) -> Optional[TransferRequest]:
        """Get status of a transfer."""
        return self.orchestrator.get_transfer_status(transfer_id)

    # -------------------------------------------------------------------------
    # Context Building
    # -------------------------------------------------------------------------

    def build_transfer_context(
        self,
        call_id: str,
        conversation_summary: str = "",
        customer_name: str = "",
        customer_id: str = "",
        issue_type: str = "",
        issue_description: str = "",
        sentiment: str = "neutral",
        pending_actions: List[str] = None,
        agent_notes: str = "",
        is_vip: bool = False,
        required_skills: List[str] = None,
        preferred_language: str = "",
        custom_data: Dict[str, Any] = None,
    ) -> TransferContext:
        """Build transfer context manually."""
        summary = ConversationSummaryContext(
            summary=conversation_summary,
            customer_name=customer_name,
            customer_id=customer_id,
            issue_type=issue_type,
            issue_description=issue_description,
            customer_sentiment=sentiment,
            pending_actions=pending_actions or [],
            agent_notes=agent_notes,
        )

        return TransferContext(
            conversation_summary=summary,
            call_id=call_id,
            is_vip=is_vip,
            required_skills=required_skills or [],
            preferred_language=preferred_language,
            custom_data=custom_data or {},
        )

    # -------------------------------------------------------------------------
    # Analytics
    # -------------------------------------------------------------------------

    def get_transfer_metrics(
        self, organization_id: str
    ) -> TransferMetrics:
        """Get transfer metrics."""
        return self.analytics.get_metrics(organization_id)

    def get_agent_transfer_metrics(
        self, agent_id: str
    ) -> AgentTransferMetrics:
        """Get agent transfer metrics."""
        return self.analytics.get_agent_metrics(agent_id)

    def get_transfer_history(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TransferResult]:
        """Get transfer history."""
        return self.orchestrator.get_transfer_history(limit, offset)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Registry
    "AgentRegistry",
    # Routing
    "RoutingEngine",
    # Orchestration
    "TransferOrchestrator",
    # Context
    "ContextBuilder",
    # Analytics
    "TransferAnalytics",
    # Main Service
    "CallTransferService",
]
