"""
Database Module

This module provides a production-ready database layer using SQLAlchemy
for the voice agent platform. It includes models, repositories, migrations,
and connection management.

Key Features:
- SQLAlchemy ORM models for all platform entities
- Repository pattern for data access
- Connection pooling and management
- Migration support with Alembic
- Transaction management
- Soft deletes and audit trails

Example usage:

    from platform.database import (
        DatabaseManager,
        get_database,
        AgentRepository,
        ConversationRepository,
        CallRepository,
    )
    from platform.database.models import Agent, Conversation, Call

    # Initialize database
    db = DatabaseManager(
        database_url="postgresql://user:pass@localhost/bvrai",
        pool_size=10,
        max_overflow=20,
    )

    # Create tables (in development)
    await db.create_all()

    # Use repositories
    async with db.session() as session:
        # Create agent
        agent_repo = AgentRepository(session)
        agent = await agent_repo.create(
            organization_id="org_123",
            name="Support Agent",
            voice_config_id="vc_456",
        )

        # Get agent
        agent = await agent_repo.get_by_id(agent.id)

        # List agents
        agents = await agent_repo.list_by_organization("org_123")

        # Update agent
        await agent_repo.update(agent.id, name="Updated Name")

        # Create conversation
        conv_repo = ConversationRepository(session)
        conversation = await conv_repo.create(
            agent_id=agent.id,
            call_id="call_789",
            customer_phone="+15551234567",
        )

        # Add message
        await conv_repo.add_message(
            conversation_id=conversation.id,
            role="user",
            content="Hello, I need help",
        )

        # Get conversation with messages
        conv = await conv_repo.get_with_messages(conversation.id)

Supported Databases:
    - PostgreSQL (recommended for production)
    - MySQL
    - SQLite (for development/testing)
"""

from .base import (
    # Base classes
    Base,
    TimestampMixin,
    SoftDeleteMixin,
    AuditMixin,
    # Database manager
    DatabaseManager,
    get_database,
    # Session management
    get_session,
    AsyncSessionFactory,
)

from .models import (
    # Organization
    Organization,
    OrganizationSettings,
    # User/API
    User,
    APIKey,
    # Agent
    Agent,
    AgentVersion,
    # Voice Config
    VoiceConfigurationModel,
    # Conversation
    Conversation,
    Message,
    # Call
    Call,
    CallEvent,
    # Analytics
    AnalyticsEvent,
    UsageRecord,
)

from .repositories import (
    # Base
    BaseRepository,
    # Repositories
    OrganizationRepository,
    UserRepository,
    AgentRepository,
    ConversationRepository,
    CallRepository,
    AnalyticsRepository,
)


__all__ = [
    # Base classes
    "Base",
    "TimestampMixin",
    "SoftDeleteMixin",
    "AuditMixin",
    # Database manager
    "DatabaseManager",
    "get_database",
    # Session management
    "get_session",
    "AsyncSessionFactory",
    # Models - Organization
    "Organization",
    "OrganizationSettings",
    # Models - User/API
    "User",
    "APIKey",
    # Models - Agent
    "Agent",
    "AgentVersion",
    # Models - Voice
    "VoiceConfigurationModel",
    # Models - Conversation
    "Conversation",
    "Message",
    # Models - Call
    "Call",
    "CallEvent",
    # Models - Analytics
    "AnalyticsEvent",
    "UsageRecord",
    # Repositories
    "BaseRepository",
    "OrganizationRepository",
    "UserRepository",
    "AgentRepository",
    "ConversationRepository",
    "CallRepository",
    "AnalyticsRepository",
]
