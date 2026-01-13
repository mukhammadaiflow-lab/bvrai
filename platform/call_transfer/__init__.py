"""
Call Transfer & Handoff Module

This module provides intelligent call transfer and handoff management
for AI voice agents, supporting warm transfers, cold transfers,
skill-based routing, and context-aware handoffs.

Key Features:
- Multiple transfer types (warm, cold, blind, consultative, conference)
- Transfer targets (human agents, departments, AI agents, external)
- Skill-based routing with rules engine
- Context passing during transfers
- Agent availability tracking
- Transfer analytics and metrics

Example usage:

    from platform.call_transfer import (
        CallTransferService,
        TransferType,
        TransferReason,
        TransferPriority,
        RoutingStrategy,
    )

    # Initialize service
    service = CallTransferService()

    # Register human agents
    agent1 = service.register_human_agent(
        name="John Smith",
        email="john@example.com",
        skills=["billing", "technical_support"],
        languages=["en", "es"],
        department="support",
        extension="1001",
    )

    agent2 = service.register_human_agent(
        name="Jane Doe",
        email="jane@example.com",
        skills=["sales", "upselling"],
        languages=["en"],
        department="sales",
        extension="2001",
    )

    # Register agent groups
    support_group = service.register_agent_group(
        name="Support Team",
        agent_ids=[agent1.id],
        skills=["support"],
        routing_strategy="least_busy",
    )

    # Register departments
    support_dept = service.register_department(
        name="Customer Support",
        agent_group_ids=[support_group.id],
        main_number="+15551234567",
    )

    # Add routing rules
    service.add_routing_rule(
        organization_id="org_123",
        name="VIP to Senior Agent",
        conditions=[
            {"type": "customer_type", "value": "vip"},
        ],
        target_type=TransferTargetType.AGENT_GROUP,
        target_ids=[support_group.id],
        strategy=RoutingStrategy.SKILL_BASED,
        priority=10,  # Higher priority (lower number)
    )

    service.add_routing_rule(
        organization_id="org_123",
        name="Billing Issues to Billing Team",
        conditions=[
            {"type": "skill_match", "value": "billing"},
        ],
        target_type=TransferTargetType.AGENT_GROUP,
        target_ids=[support_group.id],
        strategy=RoutingStrategy.LEAST_BUSY,
    )

    # Update agent availability
    service.update_agent_availability(
        agent_id=agent1.id,
        is_available=True,
        current_status="available",
    )

    # Build transfer context
    context = service.build_transfer_context(
        call_id="call_123",
        conversation_summary="Customer called about billing issue",
        customer_name="Alice Johnson",
        customer_id="cust_456",
        issue_type="billing",
        issue_description="Incorrect charge on statement",
        sentiment="negative",
        pending_actions=["Review charge", "Process refund"],
        agent_notes="Customer very frustrated, handle with care",
        is_vip=True,
        required_skills=["billing"],
    )

    # Initiate warm transfer
    result = await service.initiate_transfer(
        organization_id="org_123",
        call_id="call_123",
        transfer_type=TransferType.WARM,
        reason=TransferReason.BILLING_ISSUE,
        target_skills=["billing"],
        context=context,
        priority=TransferPriority.HIGH,
    )

    if result.success:
        print(f"Transferred to: {result.connected_name}")
        print(f"Ring time: {result.ring_time_seconds}s")
    else:
        print(f"Transfer failed: {result.error_message}")

    # Transfer to department
    result = await service.transfer_to_department(
        organization_id="org_123",
        call_id="call_456",
        department_id=support_dept.id,
        transfer_type=TransferType.COLD,
    )

    # Transfer by skills
    result = await service.transfer_to_skills(
        organization_id="org_123",
        call_id="call_789",
        skills=["technical_support", "troubleshooting"],
    )

    # Escalate to supervisor
    result = await service.escalate_call(
        organization_id="org_123",
        call_id="call_999",
        reason=TransferReason.CUSTOMER_FRUSTRATED,
        context=context,
    )

    # Get available agents
    available = service.get_available_agents(
        skills=["billing"],
        department="support",
        language="en",
    )

    # Get transfer metrics
    metrics = service.get_transfer_metrics("org_123")
    print(f"Total transfers: {metrics.total_transfers}")
    print(f"Success rate: {metrics.success_rate:.1%}")

    # Get agent metrics
    agent_metrics = service.get_agent_transfer_metrics(agent1.id)
    print(f"Transfers received: {agent_metrics.transfers_in}")

Transfer Types:
    - WARM: Agent stays until handoff complete
    - COLD: Immediate transfer, agent disconnects
    - BLIND: Transfer without announcement
    - CONSULTATIVE: Agent consults with target first
    - CONFERENCE: All parties stay connected
    - ESCALATION: Transfer to supervisor

Routing Strategies:
    - ROUND_ROBIN: Rotate through agents
    - LEAST_BUSY: Agent with fewest active calls
    - LONGEST_IDLE: Agent idle the longest
    - SKILL_BASED: Match required skills
    - PRIORITY_BASED: Based on priority rules
    - DIRECT: Direct to specific agent
"""

# Base types and enums
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

# Services
from .service import (
    # Registry
    AgentRegistry,
    # Routing
    RoutingEngine,
    # Orchestration
    TransferOrchestrator,
    # Context
    ContextBuilder,
    # Analytics
    TransferAnalytics,
    # Main Service
    CallTransferService,
)


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
    # Exceptions
    "TransferError",
    "TransferTargetNotFoundError",
    "TransferTargetUnavailableError",
    "TransferTimeoutError",
    "TransferRejectedError",
    "NoAvailableAgentsError",
    "RoutingRuleNotFoundError",
]
