"""
Queue & Routing Module

This module provides comprehensive call queue management, skill-based routing,
agent availability tracking, and overflow handling for voice agent platforms.

Features:
- Call Queues: Priority-based queuing with configurable limits
- Skill-Based Routing: Match callers to agents based on skills
- Multiple Routing Strategies: Round-robin, least-busy, skills-based, weighted
- Agent Management: Status tracking, capacity management, performance metrics
- Overflow Handling: Voicemail, callback, transfer, AI agent fallback
- Callbacks: Scheduled callback management with retry logic
- Real-Time Analytics: Queue metrics, agent performance, SLA tracking

Example usage:

    from platform.queue_routing import (
        QueueRoutingService,
        QueueConfig,
        QueuePriority,
        RoutingStrategy,
        AgentStatus,
        SkillLevel,
        OverflowAction,
    )

    # Initialize service
    service = QueueRoutingService()

    # Create skills
    spanish_skill = await service.skills.create_skill(
        organization_id="org_123",
        name="Spanish",
        category="language",
    )

    technical_skill = await service.skills.create_skill(
        organization_id="org_123",
        name="Technical Support",
        category="support",
    )

    # Create a queue with skill requirements
    queue = await service.queues.create_queue(
        organization_id="org_123",
        name="Technical Support - Spanish",
        config=QueueConfig(
            routing_strategy=RoutingStrategy.SKILLS_BASED,
            max_wait_time_seconds=300,
            target_answer_time_seconds=30,
            overflow_action=OverflowAction.CALLBACK,
        ),
        skill_requirements=[
            SkillRequirement(
                skill_id=spanish_skill.id,
                skill_name="Spanish",
                minimum_level=SkillLevel.INTERMEDIATE,
                required=True,
            ),
            SkillRequirement(
                skill_id=technical_skill.id,
                skill_name="Technical Support",
                minimum_level=SkillLevel.ADVANCED,
                required=True,
            ),
        ],
    )

    # Create agents with skills
    agent = await service.agents.create_agent(
        organization_id="org_123",
        name="Maria Garcia",
        email="maria@example.com",
        skills=[
            AgentSkill(
                skill_id=spanish_skill.id,
                skill_name="Spanish",
                level=SkillLevel.EXPERT,
                certified=True,
            ),
            AgentSkill(
                skill_id=technical_skill.id,
                skill_name="Technical Support",
                level=SkillLevel.ADVANCED,
            ),
        ],
        queue_ids=[queue.id],
    )

    # Set agent as available
    await service.agents.set_agent_status(agent.id, AgentStatus.AVAILABLE)

    # Enqueue a call
    entry, decision = await service.enqueue_call(
        queue_id=queue.id,
        call_id="call_456",
        caller_phone="+15551234567",
        caller_name="John Doe",
        priority=QueuePriority.HIGH,
        is_vip=True,
        route_immediately=True,
    )

    if decision.routed:
        print(f"Call routed to agent {decision.agent_id}")
        print(f"Match score: {decision.match_score}")

    # Complete the call
    await service.complete_call(
        entry_id=entry.id,
        agent_id=decision.agent_id,
        handle_time_seconds=300,
        wrap_up_time_seconds=30,
    )

    # Get queue metrics
    metrics = await service.analytics.get_queue_metrics(queue.id)
    print(f"Calls answered: {metrics.calls_answered}")
    print(f"SLA: {metrics.service_level_percent}%")

    # Schedule a callback
    callback = await service.schedule_callback(
        entry_id=entry.id,
        preferred_time=datetime.utcnow() + timedelta(hours=1),
    )

    # Get organization dashboard
    dashboard = await service.get_organization_dashboard("org_123")
"""

# Base types and enums
from .base import (
    # Enums
    QueueStatus,
    QueuePriority,
    RoutingStrategy,
    AgentStatus,
    OverflowAction,
    CallOutcome,
    SkillLevel,
    # Skill types
    Skill,
    AgentSkill,
    SkillRequirement,
    # Agent types
    QueueAgent,
    # Queue types
    QueueConfig,
    CallQueue,
    QueueEntry,
    # Routing types
    RoutingDecision,
    CallbackRequest,
    # Analytics types
    QueueMetrics,
    AgentMetrics,
    # Exceptions
    QueueRoutingError,
    QueueFullError,
    QueueNotFoundError,
    AgentNotFoundError,
    NoAgentAvailableError,
    SkillMismatchError,
    QueueClosedError,
)

# Services
from .service import (
    # Managers
    SkillManager,
    AgentManager,
    QueueManager,
    # Routing strategies
    RoundRobinStrategy,
    LeastBusyStrategy,
    LongestIdleStrategy,
    SkillsBasedStrategy,
    WeightedStrategy,
    StickyStrategy,
    RoutingEngine,
    # Callbacks
    CallbackManager,
    # Analytics
    QueueAnalyticsService,
    # Main Service
    QueueRoutingService,
)


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
    # Managers
    "SkillManager",
    "AgentManager",
    "QueueManager",
    # Routing strategies
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
    # Exceptions
    "QueueRoutingError",
    "QueueFullError",
    "QueueNotFoundError",
    "AgentNotFoundError",
    "NoAgentAvailableError",
    "SkillMismatchError",
    "QueueClosedError",
]
