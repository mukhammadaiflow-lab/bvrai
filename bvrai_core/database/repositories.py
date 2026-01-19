"""
Database Repositories

Repository pattern implementation for data access.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from sqlalchemy import select, update, delete, func, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .base import Base
from .models import (
    Organization,
    OrganizationSettings,
    User,
    APIKey,
    Agent,
    AgentVersion,
    VoiceConfigurationModel,
    Conversation,
    Message,
    Call,
    CallEvent,
    AnalyticsEvent,
    UsageRecord,
)


# =============================================================================
# Generic Type Variable
# =============================================================================


ModelType = TypeVar("ModelType", bound=Base)


# =============================================================================
# Base Repository
# =============================================================================


class BaseRepository(Generic[ModelType]):
    """Base repository with common CRUD operations."""

    model: Type[ModelType]

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, id: str) -> Optional[ModelType]:
        """Get entity by ID."""
        result = await self.session.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalar_one_or_none()

    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[ModelType]:
        """Get all entities with pagination."""
        result = await self.session.execute(
            select(self.model)
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def create(self, **kwargs) -> ModelType:
        """Create a new entity."""
        instance = self.model(**kwargs)
        self.session.add(instance)
        await self.session.flush()
        await self.session.refresh(instance)
        return instance

    async def update(self, id: str, **kwargs) -> Optional[ModelType]:
        """Update an entity."""
        instance = await self.get_by_id(id)
        if not instance:
            return None

        for key, value in kwargs.items():
            if hasattr(instance, key):
                setattr(instance, key, value)

        await self.session.flush()
        await self.session.refresh(instance)
        return instance

    async def delete(self, id: str) -> bool:
        """Delete an entity."""
        instance = await self.get_by_id(id)
        if not instance:
            return False

        await self.session.delete(instance)
        return True

    async def soft_delete(self, id: str) -> bool:
        """Soft delete an entity."""
        instance = await self.get_by_id(id)
        if not instance:
            return False

        if hasattr(instance, "soft_delete"):
            instance.soft_delete()
            await self.session.flush()
            return True
        return False

    async def count(self) -> int:
        """Count all entities."""
        result = await self.session.execute(
            select(func.count()).select_from(self.model)
        )
        return result.scalar_one()

    async def exists(self, id: str) -> bool:
        """Check if entity exists."""
        result = await self.session.execute(
            select(func.count())
            .select_from(self.model)
            .where(self.model.id == id)
        )
        return result.scalar_one() > 0


# =============================================================================
# Organization Repository
# =============================================================================


class OrganizationRepository(BaseRepository[Organization]):
    """Repository for Organization entities."""

    model = Organization

    async def get_by_slug(self, slug: str) -> Optional[Organization]:
        """Get organization by slug."""
        result = await self.session.execute(
            select(Organization).where(Organization.slug == slug)
        )
        return result.scalar_one_or_none()

    async def get_with_settings(self, id: str) -> Optional[Organization]:
        """Get organization with settings."""
        result = await self.session.execute(
            select(Organization)
            .options(selectinload(Organization.settings))
            .where(Organization.id == id)
        )
        return result.scalar_one_or_none()

    async def list_active(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Organization]:
        """List active organizations."""
        result = await self.session.execute(
            select(Organization)
            .where(
                and_(
                    Organization.is_active == True,
                    Organization.is_deleted == False,
                )
            )
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def search(
        self,
        query: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Organization]:
        """Search organizations by name."""
        result = await self.session.execute(
            select(Organization)
            .where(
                and_(
                    Organization.name.ilike(f"%{query}%"),
                    Organization.is_deleted == False,
                )
            )
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())


# =============================================================================
# User Repository
# =============================================================================


class UserRepository(BaseRepository[User]):
    """Repository for User entities."""

    model = User

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()

    async def list_by_organization(
        self,
        organization_id: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[User]:
        """List users by organization."""
        result = await self.session.execute(
            select(User)
            .where(
                and_(
                    User.organization_id == organization_id,
                    User.is_deleted == False,
                )
            )
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count_by_organization(self, organization_id: str) -> int:
        """Count users in organization."""
        result = await self.session.execute(
            select(func.count())
            .select_from(User)
            .where(
                and_(
                    User.organization_id == organization_id,
                    User.is_deleted == False,
                )
            )
        )
        return result.scalar_one()

    async def update_last_login(self, id: str) -> None:
        """Update last login timestamp."""
        await self.session.execute(
            update(User)
            .where(User.id == id)
            .values(last_login_at=datetime.utcnow())
        )


# =============================================================================
# Agent Repository
# =============================================================================


class AgentRepository(BaseRepository[Agent]):
    """Repository for Agent entities."""

    model = Agent

    async def list_by_organization(
        self,
        organization_id: str,
        include_inactive: bool = False,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Agent]:
        """List agents by organization."""
        conditions = [
            Agent.organization_id == organization_id,
            Agent.is_deleted == False,
        ]
        if not include_inactive:
            conditions.append(Agent.is_active == True)

        result = await self.session.execute(
            select(Agent)
            .where(and_(*conditions))
            .order_by(Agent.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_slug(
        self,
        organization_id: str,
        slug: str,
    ) -> Optional[Agent]:
        """Get agent by slug within organization."""
        result = await self.session.execute(
            select(Agent)
            .where(
                and_(
                    Agent.organization_id == organization_id,
                    Agent.slug == slug,
                    Agent.is_deleted == False,
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_with_voice_config(self, id: str) -> Optional[Agent]:
        """Get agent with voice configuration."""
        result = await self.session.execute(
            select(Agent)
            .options(selectinload(Agent.voice_config))
            .where(Agent.id == id)
        )
        return result.scalar_one_or_none()

    async def increment_usage(
        self,
        id: str,
        calls: int = 0,
        minutes: float = 0.0,
    ) -> None:
        """Increment agent usage stats."""
        await self.session.execute(
            update(Agent)
            .where(Agent.id == id)
            .values(
                total_calls=Agent.total_calls + calls,
                total_minutes=Agent.total_minutes + minutes,
            )
        )

    async def create_version(
        self,
        agent_id: str,
        change_notes: str = "",
        created_by_user_id: str = None,
    ) -> AgentVersion:
        """Create a new version snapshot."""
        agent = await self.get_by_id(agent_id)
        if not agent:
            raise ValueError(f"Agent not found: {agent_id}")

        version = AgentVersion(
            agent_id=agent_id,
            version_number=agent.current_version + 1,
            system_prompt=agent.system_prompt,
            first_message=agent.first_message,
            llm_provider=agent.llm_provider,
            llm_model=agent.llm_model,
            llm_temperature=agent.llm_temperature,
            config_snapshot={
                "llm_max_tokens": agent.llm_max_tokens,
                "voice_config_id": agent.voice_config_id,
                "metadata": agent.metadata_json,
            },
            change_notes=change_notes,
            created_by_user_id=created_by_user_id,
        )

        self.session.add(version)
        agent.current_version += 1

        await self.session.flush()
        return version


# =============================================================================
# Conversation Repository
# =============================================================================


class ConversationRepository(BaseRepository[Conversation]):
    """Repository for Conversation entities."""

    model = Conversation

    async def list_by_organization(
        self,
        organization_id: str,
        status: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Conversation]:
        """List conversations by organization with filters."""
        conditions = [
            Conversation.organization_id == organization_id,
            Conversation.is_deleted == False,
        ]

        if status:
            conditions.append(Conversation.status == status)
        if start_date:
            conditions.append(Conversation.started_at >= start_date)
        if end_date:
            conditions.append(Conversation.started_at <= end_date)

        result = await self.session.execute(
            select(Conversation)
            .where(and_(*conditions))
            .order_by(Conversation.started_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def list_by_agent(
        self,
        agent_id: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Conversation]:
        """List conversations by agent."""
        result = await self.session.execute(
            select(Conversation)
            .where(
                and_(
                    Conversation.agent_id == agent_id,
                    Conversation.is_deleted == False,
                )
            )
            .order_by(Conversation.started_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_with_messages(self, id: str) -> Optional[Conversation]:
        """Get conversation with messages."""
        result = await self.session.execute(
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == id)
        )
        return result.scalar_one_or_none()

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        **kwargs,
    ) -> Message:
        """Add a message to conversation."""
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            **kwargs,
        )
        self.session.add(message)

        # Update conversation stats
        await self.session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(
                message_count=Conversation.message_count + 1,
                updated_at=datetime.utcnow(),
            )
        )

        await self.session.flush()
        return message

    async def end_conversation(
        self,
        id: str,
        outcome: str = None,
        resolution: str = None,
    ) -> Optional[Conversation]:
        """End a conversation."""
        conversation = await self.get_by_id(id)
        if not conversation:
            return None

        conversation.status = "completed"
        conversation.ended_at = datetime.utcnow()
        conversation.duration_seconds = (
            conversation.ended_at - conversation.started_at
        ).total_seconds()

        if outcome:
            conversation.outcome = outcome
        if resolution:
            conversation.resolution = resolution

        await self.session.flush()
        return conversation

    async def get_recent_by_customer(
        self,
        customer_phone: str,
        limit: int = 10,
    ) -> List[Conversation]:
        """Get recent conversations by customer phone."""
        result = await self.session.execute(
            select(Conversation)
            .where(
                and_(
                    Conversation.customer_phone == customer_phone,
                    Conversation.is_deleted == False,
                )
            )
            .order_by(Conversation.started_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


# =============================================================================
# Call Repository
# =============================================================================


class CallRepository(BaseRepository[Call]):
    """Repository for Call entities."""

    model = Call

    async def list_by_organization(
        self,
        organization_id: str,
        status: str = None,
        direction: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Call]:
        """List calls by organization with filters."""
        conditions = [
            Call.organization_id == organization_id,
            Call.is_deleted == False,
        ]

        if status:
            conditions.append(Call.status == status)
        if direction:
            conditions.append(Call.direction == direction)
        if start_date:
            conditions.append(Call.initiated_at >= start_date)
        if end_date:
            conditions.append(Call.initiated_at <= end_date)

        result = await self.session.execute(
            select(Call)
            .where(and_(*conditions))
            .order_by(Call.initiated_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_external_id(
        self,
        external_call_id: str,
    ) -> Optional[Call]:
        """Get call by external ID."""
        result = await self.session.execute(
            select(Call).where(Call.external_call_id == external_call_id)
        )
        return result.scalar_one_or_none()

    async def get_with_events(self, id: str) -> Optional[Call]:
        """Get call with events."""
        result = await self.session.execute(
            select(Call)
            .options(selectinload(Call.events))
            .where(Call.id == id)
        )
        return result.scalar_one_or_none()

    async def add_event(
        self,
        call_id: str,
        event_type: str,
        event_data: Dict = None,
    ) -> CallEvent:
        """Add an event to call."""
        call = await self.get_by_id(call_id)
        relative_time = 0
        if call and call.initiated_at:
            relative_time = int(
                (datetime.utcnow() - call.initiated_at).total_seconds() * 1000
            )

        event = CallEvent(
            call_id=call_id,
            event_type=event_type,
            event_data=event_data,
            relative_time_ms=relative_time,
        )
        self.session.add(event)
        await self.session.flush()
        return event

    async def update_status(
        self,
        id: str,
        status: str,
        end_reason: str = None,
    ) -> Optional[Call]:
        """Update call status."""
        call = await self.get_by_id(id)
        if not call:
            return None

        call.status = status

        if status == "in_progress" and not call.answered_at:
            call.answered_at = datetime.utcnow()
            call.ring_duration_seconds = (
                call.answered_at - call.initiated_at
            ).total_seconds()

        if status in ["completed", "failed", "no_answer", "busy", "cancelled"]:
            call.ended_at = datetime.utcnow()
            if call.answered_at:
                call.duration_seconds = (
                    call.ended_at - call.answered_at
                ).total_seconds()
            if end_reason:
                call.end_reason = end_reason

        await self.session.flush()
        return call

    async def get_daily_stats(
        self,
        organization_id: str,
        date: datetime,
    ) -> Dict[str, Any]:
        """Get daily call statistics."""
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)

        # Total calls
        total_result = await self.session.execute(
            select(func.count())
            .select_from(Call)
            .where(
                and_(
                    Call.organization_id == organization_id,
                    Call.initiated_at >= start,
                    Call.initiated_at < end,
                )
            )
        )
        total = total_result.scalar_one()

        # Completed calls
        completed_result = await self.session.execute(
            select(func.count())
            .select_from(Call)
            .where(
                and_(
                    Call.organization_id == organization_id,
                    Call.initiated_at >= start,
                    Call.initiated_at < end,
                    Call.status == "completed",
                )
            )
        )
        completed = completed_result.scalar_one()

        # Total duration
        duration_result = await self.session.execute(
            select(func.sum(Call.duration_seconds))
            .where(
                and_(
                    Call.organization_id == organization_id,
                    Call.initiated_at >= start,
                    Call.initiated_at < end,
                )
            )
        )
        total_duration = duration_result.scalar_one() or 0

        return {
            "date": date.date().isoformat(),
            "total_calls": total,
            "completed_calls": completed,
            "total_duration_seconds": total_duration,
            "average_duration_seconds": total_duration / max(completed, 1),
        }


# =============================================================================
# Analytics Repository
# =============================================================================


class AnalyticsRepository(BaseRepository[AnalyticsEvent]):
    """Repository for Analytics entities."""

    model = AnalyticsEvent

    async def track_event(
        self,
        organization_id: str,
        event_type: str,
        event_name: str,
        properties: Dict = None,
        agent_id: str = None,
        user_id: str = None,
        call_id: str = None,
        conversation_id: str = None,
        source: str = "system",
    ) -> AnalyticsEvent:
        """Track an analytics event."""
        event = AnalyticsEvent(
            organization_id=organization_id,
            event_type=event_type,
            event_name=event_name,
            properties=properties,
            agent_id=agent_id,
            user_id=user_id,
            call_id=call_id,
            conversation_id=conversation_id,
            source=source,
        )
        self.session.add(event)
        await self.session.flush()
        return event

    async def list_by_organization(
        self,
        organization_id: str,
        event_type: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[AnalyticsEvent]:
        """List events by organization."""
        conditions = [AnalyticsEvent.organization_id == organization_id]

        if event_type:
            conditions.append(AnalyticsEvent.event_type == event_type)
        if start_date:
            conditions.append(AnalyticsEvent.created_at >= start_date)
        if end_date:
            conditions.append(AnalyticsEvent.created_at <= end_date)

        result = await self.session.execute(
            select(AnalyticsEvent)
            .where(and_(*conditions))
            .order_by(AnalyticsEvent.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count_by_type(
        self,
        organization_id: str,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> Dict[str, int]:
        """Count events by type."""
        conditions = [AnalyticsEvent.organization_id == organization_id]

        if start_date:
            conditions.append(AnalyticsEvent.created_at >= start_date)
        if end_date:
            conditions.append(AnalyticsEvent.created_at <= end_date)

        result = await self.session.execute(
            select(
                AnalyticsEvent.event_type,
                func.count().label("count"),
            )
            .where(and_(*conditions))
            .group_by(AnalyticsEvent.event_type)
        )

        return {row.event_type: row.count for row in result.all()}

    async def record_usage(
        self,
        organization_id: str,
        usage_type: str,
        quantity: float,
        unit: str,
        period_start: datetime,
        period_end: datetime,
        agent_id: str = None,
        unit_cost: float = 0.0,
    ) -> UsageRecord:
        """Record usage for billing."""
        record = UsageRecord(
            organization_id=organization_id,
            usage_type=usage_type,
            quantity=quantity,
            unit=unit,
            period_start=period_start,
            period_end=period_end,
            agent_id=agent_id,
            unit_cost=unit_cost,
            total_cost=quantity * unit_cost,
        )
        self.session.add(record)
        await self.session.flush()
        return record

    async def get_usage_summary(
        self,
        organization_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """Get usage summary for a period."""
        result = await self.session.execute(
            select(
                UsageRecord.usage_type,
                func.sum(UsageRecord.quantity).label("total_quantity"),
                func.sum(UsageRecord.total_cost).label("total_cost"),
            )
            .where(
                and_(
                    UsageRecord.organization_id == organization_id,
                    UsageRecord.period_start >= start_date,
                    UsageRecord.period_end <= end_date,
                )
            )
            .group_by(UsageRecord.usage_type)
        )

        summary = {}
        total_cost = 0.0

        for row in result.all():
            summary[row.usage_type] = {
                "quantity": row.total_quantity,
                "cost": row.total_cost,
            }
            total_cost += row.total_cost or 0

        return {
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "by_type": summary,
            "total_cost": total_cost,
        }


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "BaseRepository",
    "OrganizationRepository",
    "UserRepository",
    "AgentRepository",
    "ConversationRepository",
    "CallRepository",
    "AnalyticsRepository",
]
