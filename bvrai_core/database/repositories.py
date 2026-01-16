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
    PhoneNumber,
    Webhook,
    WebhookDelivery,
    KnowledgeBase,
    Document,
    DocumentChunk,
    Campaign,
    CampaignContact,
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

    async def get_by_organization(
        self,
        organization_id: str,
        skip: int = 0,
        limit: int = 100,
        is_active: bool = None,
    ) -> List[Agent]:
        """Get agents by organization with filters."""
        conditions = [
            Agent.organization_id == organization_id,
            Agent.is_deleted == False,
        ]
        if is_active is not None:
            conditions.append(Agent.is_active == is_active)

        result = await self.session.execute(
            select(Agent)
            .where(and_(*conditions))
            .order_by(Agent.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count_by_organization(
        self,
        organization_id: str,
        is_active: bool = None,
    ) -> int:
        """Count agents by organization."""
        conditions = [
            Agent.organization_id == organization_id,
            Agent.is_deleted == False,
        ]
        if is_active is not None:
            conditions.append(Agent.is_active == is_active)

        result = await self.session.execute(
            select(func.count())
            .select_from(Agent)
            .where(and_(*conditions))
        )
        return result.scalar_one()

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

        # Get current version number
        current_version = getattr(agent, 'current_version', 0) or 0

        version = AgentVersion(
            agent_id=agent_id,
            version_number=current_version + 1,
            config_snapshot={
                "system_prompt": agent.system_prompt,
                "first_message": agent.first_message,
                "voice_config": agent.voice_config,
                "llm_config": agent.llm_config,
                "behavior_config": agent.behavior_config,
                "transcription_config": agent.transcription_config,
                "knowledge_base_ids": agent.knowledge_base_ids,
                "functions": agent.functions,
                "metadata": agent.metadata,
            },
            change_notes=change_notes,
            created_by=created_by_user_id,
        )

        self.session.add(version)

        # Update agent version
        if hasattr(agent, 'current_version'):
            agent.current_version = current_version + 1

        await self.session.flush()
        return version

    async def get_versions(
        self,
        agent_id: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[AgentVersion]:
        """Get version history for an agent."""
        result = await self.session.execute(
            select(AgentVersion)
            .where(AgentVersion.agent_id == agent_id)
            .order_by(AgentVersion.version_number.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def rollback_to_version(
        self,
        agent_id: str,
        version_id: str,
    ) -> Optional[Agent]:
        """Rollback agent to a previous version."""
        # Get the version
        version_result = await self.session.execute(
            select(AgentVersion)
            .where(
                and_(
                    AgentVersion.id == version_id,
                    AgentVersion.agent_id == agent_id,
                )
            )
        )
        version = version_result.scalar_one_or_none()

        if not version:
            return None

        agent = await self.get_by_id(agent_id)
        if not agent:
            return None

        # Restore from snapshot
        snapshot = version.config_snapshot or {}
        if "system_prompt" in snapshot:
            agent.system_prompt = snapshot["system_prompt"]
        if "first_message" in snapshot:
            agent.first_message = snapshot["first_message"]
        if "voice_config" in snapshot:
            agent.voice_config = snapshot["voice_config"]
        if "llm_config" in snapshot:
            agent.llm_config = snapshot["llm_config"]
        if "behavior_config" in snapshot:
            agent.behavior_config = snapshot["behavior_config"]
        if "transcription_config" in snapshot:
            agent.transcription_config = snapshot["transcription_config"]
        if "knowledge_base_ids" in snapshot:
            agent.knowledge_base_ids = snapshot["knowledge_base_ids"]
        if "functions" in snapshot:
            agent.functions = snapshot["functions"]
        if "metadata" in snapshot:
            agent.metadata = snapshot["metadata"]

        await self.session.flush()
        await self.session.refresh(agent)
        return agent


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
        agent_id: str = None,
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

        if agent_id:
            conditions.append(Call.agent_id == agent_id)
        if status:
            conditions.append(Call.status == status)
        if direction:
            conditions.append(Call.direction == direction)
        if start_date:
            conditions.append(Call.created_at >= start_date)
        if end_date:
            conditions.append(Call.created_at <= end_date)

        result = await self.session.execute(
            select(Call)
            .where(and_(*conditions))
            .order_by(Call.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count_by_organization(
        self,
        organization_id: str,
        agent_id: str = None,
        status: str = None,
    ) -> int:
        """Count calls by organization."""
        conditions = [
            Call.organization_id == organization_id,
            Call.is_deleted == False,
        ]

        if agent_id:
            conditions.append(Call.agent_id == agent_id)
        if status:
            conditions.append(Call.status == status)

        result = await self.session.execute(
            select(func.count())
            .select_from(Call)
            .where(and_(*conditions))
        )
        return result.scalar_one()

    async def get_events(
        self,
        call_id: str,
        event_type: str = None,
    ) -> List[CallEvent]:
        """Get events for a call."""
        conditions = [CallEvent.call_id == call_id]
        if event_type:
            conditions.append(CallEvent.event_type == event_type)

        result = await self.session.execute(
            select(CallEvent)
            .where(and_(*conditions))
            .order_by(CallEvent.relative_time_ms)
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
# Phone Number Repository
# =============================================================================


class PhoneNumberRepository(BaseRepository[PhoneNumber]):
    """Repository for PhoneNumber entities."""

    model = PhoneNumber

    async def get_by_number(self, number: str) -> Optional[PhoneNumber]:
        """Get phone number by E.164 number."""
        result = await self.session.execute(
            select(PhoneNumber).where(
                and_(
                    PhoneNumber.number == number,
                    PhoneNumber.is_deleted == False,
                )
            )
        )
        return result.scalar_one_or_none()

    async def list_by_organization(
        self,
        organization_id: str,
        status: str = None,
        agent_id: str = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[PhoneNumber]:
        """List phone numbers by organization."""
        conditions = [
            PhoneNumber.organization_id == organization_id,
            PhoneNumber.is_deleted == False,
        ]
        if status:
            conditions.append(PhoneNumber.status == status)
        if agent_id:
            conditions.append(PhoneNumber.agent_id == agent_id)

        result = await self.session.execute(
            select(PhoneNumber)
            .where(and_(*conditions))
            .order_by(PhoneNumber.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count_by_organization(
        self,
        organization_id: str,
        status: str = None,
    ) -> int:
        """Count phone numbers by organization."""
        conditions = [
            PhoneNumber.organization_id == organization_id,
            PhoneNumber.is_deleted == False,
        ]
        if status:
            conditions.append(PhoneNumber.status == status)

        result = await self.session.execute(
            select(func.count())
            .select_from(PhoneNumber)
            .where(and_(*conditions))
        )
        return result.scalar_one()

    async def get_available(
        self,
        organization_id: str,
    ) -> List[PhoneNumber]:
        """Get available (unassigned) phone numbers."""
        result = await self.session.execute(
            select(PhoneNumber).where(
                and_(
                    PhoneNumber.organization_id == organization_id,
                    PhoneNumber.agent_id == None,
                    PhoneNumber.status == "active",
                    PhoneNumber.is_deleted == False,
                )
            )
        )
        return list(result.scalars().all())

    async def assign_to_agent(
        self,
        phone_number_id: str,
        agent_id: str,
    ) -> Optional[PhoneNumber]:
        """Assign phone number to an agent."""
        phone = await self.get_by_id(phone_number_id)
        if not phone:
            return None
        phone.agent_id = agent_id
        await self.session.flush()
        await self.session.refresh(phone)
        return phone

    async def unassign_from_agent(self, phone_number_id: str) -> Optional[PhoneNumber]:
        """Unassign phone number from agent."""
        return await self.assign_to_agent(phone_number_id, None)


# =============================================================================
# Webhook Repository
# =============================================================================


class WebhookRepository(BaseRepository[Webhook]):
    """Repository for Webhook entities."""

    model = Webhook

    async def list_by_organization(
        self,
        organization_id: str,
        is_active: bool = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Webhook]:
        """List webhooks by organization."""
        conditions = [
            Webhook.organization_id == organization_id,
            Webhook.is_deleted == False,
        ]
        if is_active is not None:
            conditions.append(Webhook.is_active == is_active)

        result = await self.session.execute(
            select(Webhook)
            .where(and_(*conditions))
            .order_by(Webhook.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count_by_organization(
        self,
        organization_id: str,
        is_active: bool = None,
    ) -> int:
        """Count webhooks by organization."""
        conditions = [
            Webhook.organization_id == organization_id,
            Webhook.is_deleted == False,
        ]
        if is_active is not None:
            conditions.append(Webhook.is_active == is_active)

        result = await self.session.execute(
            select(func.count())
            .select_from(Webhook)
            .where(and_(*conditions))
        )
        return result.scalar_one()

    async def get_for_event(
        self,
        organization_id: str,
        event_type: str,
        agent_id: str = None,
    ) -> List[Webhook]:
        """Get active webhooks that subscribe to an event type."""
        result = await self.session.execute(
            select(Webhook).where(
                and_(
                    Webhook.organization_id == organization_id,
                    Webhook.is_active == True,
                    Webhook.is_deleted == False,
                )
            )
        )
        webhooks = result.scalars().all()

        # Filter by event type and agent_id
        matching = []
        for webhook in webhooks:
            events = webhook.events or []
            if event_type in events or "*" in events:
                # Check agent filter
                agent_ids = webhook.agent_ids or []
                if not agent_ids or agent_id in agent_ids:
                    matching.append(webhook)

        return matching

    async def create_delivery(
        self,
        webhook_id: str,
        event_type: str,
        event_id: str,
        request_url: str,
        request_headers: Dict = None,
        request_body: Dict = None,
    ) -> WebhookDelivery:
        """Create a webhook delivery record."""
        delivery = WebhookDelivery(
            webhook_id=webhook_id,
            event_type=event_type,
            event_id=event_id,
            request_url=request_url,
            request_headers=request_headers,
            request_body=request_body,
            status="pending",
        )
        self.session.add(delivery)
        await self.session.flush()
        await self.session.refresh(delivery)
        return delivery

    async def update_delivery(
        self,
        delivery_id: str,
        status: str,
        response_status: int = None,
        response_headers: Dict = None,
        response_body: str = None,
        duration_ms: int = None,
        error_message: str = None,
    ) -> Optional[WebhookDelivery]:
        """Update webhook delivery status."""
        result = await self.session.execute(
            select(WebhookDelivery).where(WebhookDelivery.id == delivery_id)
        )
        delivery = result.scalar_one_or_none()
        if not delivery:
            return None

        delivery.status = status
        if response_status is not None:
            delivery.response_status = response_status
        if response_headers is not None:
            delivery.response_headers = response_headers
        if response_body is not None:
            delivery.response_body = response_body
        if duration_ms is not None:
            delivery.duration_ms = duration_ms
        if error_message is not None:
            delivery.error_message = error_message

        # Update webhook stats
        webhook = await self.get_by_id(delivery.webhook_id)
        if webhook:
            webhook.total_deliveries += 1
            webhook.last_triggered_at = datetime.utcnow()
            if status == "success":
                webhook.successful_deliveries += 1
                webhook.last_success_at = datetime.utcnow()
            elif status == "failed":
                webhook.failed_deliveries += 1
                webhook.last_failure_at = datetime.utcnow()

        await self.session.flush()
        await self.session.refresh(delivery)
        return delivery

    async def get_deliveries(
        self,
        webhook_id: str,
        status: str = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[WebhookDelivery]:
        """Get deliveries for a webhook."""
        conditions = [WebhookDelivery.webhook_id == webhook_id]
        if status:
            conditions.append(WebhookDelivery.status == status)

        result = await self.session.execute(
            select(WebhookDelivery)
            .where(and_(*conditions))
            .order_by(WebhookDelivery.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())


# =============================================================================
# Knowledge Base Repository
# =============================================================================


class KnowledgeBaseRepository(BaseRepository[KnowledgeBase]):
    """Repository for KnowledgeBase entities."""

    model = KnowledgeBase

    async def list_by_organization(
        self,
        organization_id: str,
        status: str = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[KnowledgeBase]:
        """List knowledge bases by organization."""
        conditions = [
            KnowledgeBase.organization_id == organization_id,
            KnowledgeBase.is_deleted == False,
        ]
        if status:
            conditions.append(KnowledgeBase.status == status)

        result = await self.session.execute(
            select(KnowledgeBase)
            .where(and_(*conditions))
            .order_by(KnowledgeBase.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count_by_organization(
        self,
        organization_id: str,
        status: str = None,
    ) -> int:
        """Count knowledge bases by organization."""
        conditions = [
            KnowledgeBase.organization_id == organization_id,
            KnowledgeBase.is_deleted == False,
        ]
        if status:
            conditions.append(KnowledgeBase.status == status)

        result = await self.session.execute(
            select(func.count())
            .select_from(KnowledgeBase)
            .where(and_(*conditions))
        )
        return result.scalar_one()

    async def get_with_documents(self, id: str) -> Optional[KnowledgeBase]:
        """Get knowledge base with documents."""
        result = await self.session.execute(
            select(KnowledgeBase)
            .options(selectinload(KnowledgeBase.documents))
            .where(KnowledgeBase.id == id)
        )
        return result.scalar_one_or_none()

    async def add_document(
        self,
        knowledge_base_id: str,
        organization_id: str,
        name: str,
        doc_type: str,
        content: str = None,
        source_url: str = None,
        file_path: str = None,
        file_size: int = None,
        mime_type: str = None,
        extra_data: Dict = None,
    ) -> Document:
        """Add a document to knowledge base."""
        document = Document(
            knowledge_base_id=knowledge_base_id,
            organization_id=organization_id,
            name=name,
            doc_type=doc_type,
            content=content,
            source_url=source_url,
            file_path=file_path,
            file_size=file_size,
            mime_type=mime_type,
            extra_data=extra_data,
            status="pending",
        )
        self.session.add(document)

        # Update document count
        await self.session.execute(
            update(KnowledgeBase)
            .where(KnowledgeBase.id == knowledge_base_id)
            .values(document_count=KnowledgeBase.document_count + 1)
        )

        await self.session.flush()
        await self.session.refresh(document)
        return document

    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID."""
        result = await self.session.execute(
            select(Document).where(Document.id == document_id)
        )
        return result.scalar_one_or_none()

    async def list_documents(
        self,
        knowledge_base_id: str,
        status: str = None,
        doc_type: str = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Document]:
        """List documents in knowledge base."""
        conditions = [
            Document.knowledge_base_id == knowledge_base_id,
            Document.is_deleted == False,
        ]
        if status:
            conditions.append(Document.status == status)
        if doc_type:
            conditions.append(Document.doc_type == doc_type)

        result = await self.session.execute(
            select(Document)
            .where(and_(*conditions))
            .order_by(Document.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def update_document_status(
        self,
        document_id: str,
        status: str,
        error_message: str = None,
        chunk_count: int = None,
        token_count: int = None,
    ) -> Optional[Document]:
        """Update document processing status."""
        result = await self.session.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        if not document:
            return None

        document.status = status
        if error_message is not None:
            document.error_message = error_message
        if chunk_count is not None:
            document.chunk_count = chunk_count
        if token_count is not None:
            document.token_count = token_count
        if status == "completed":
            document.processed_at = datetime.utcnow()

        await self.session.flush()
        await self.session.refresh(document)
        return document

    async def delete_document(self, document_id: str) -> bool:
        """Soft delete a document."""
        result = await self.session.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        if not document:
            return False

        # Soft delete
        document.is_deleted = True
        document.deleted_at = datetime.utcnow()

        # Update knowledge base counts
        await self.session.execute(
            update(KnowledgeBase)
            .where(KnowledgeBase.id == document.knowledge_base_id)
            .values(
                document_count=KnowledgeBase.document_count - 1,
                chunk_count=KnowledgeBase.chunk_count - document.chunk_count,
                total_tokens=KnowledgeBase.total_tokens - document.token_count,
            )
        )

        await self.session.flush()
        return True

    async def add_chunk(
        self,
        document_id: str,
        knowledge_base_id: str,
        content: str,
        chunk_index: int,
        start_char: int = None,
        end_char: int = None,
        token_count: int = 0,
        vector_id: str = None,
        embedding_model: str = None,
        chunk_metadata: Dict = None,
    ) -> DocumentChunk:
        """Add a chunk to a document."""
        chunk = DocumentChunk(
            document_id=document_id,
            knowledge_base_id=knowledge_base_id,
            content=content,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            token_count=token_count,
            vector_id=vector_id,
            embedding_model=embedding_model,
            chunk_metadata=chunk_metadata,
        )
        self.session.add(chunk)
        await self.session.flush()
        await self.session.refresh(chunk)
        return chunk

    async def get_chunks(
        self,
        document_id: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[DocumentChunk]:
        """Get chunks for a document."""
        result = await self.session.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())


# =============================================================================
# Campaign Repository
# =============================================================================


class CampaignRepository(BaseRepository[Campaign]):
    """Repository for Campaign entities."""

    model = Campaign

    async def list_by_organization(
        self,
        organization_id: str,
        status: str = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Campaign]:
        """List campaigns by organization."""
        conditions = [
            Campaign.organization_id == organization_id,
            Campaign.is_deleted == False,
        ]
        if status:
            conditions.append(Campaign.status == status)

        result = await self.session.execute(
            select(Campaign)
            .where(and_(*conditions))
            .order_by(Campaign.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count_by_organization(
        self,
        organization_id: str,
        status: str = None,
    ) -> int:
        """Count campaigns by organization."""
        conditions = [
            Campaign.organization_id == organization_id,
            Campaign.is_deleted == False,
        ]
        if status:
            conditions.append(Campaign.status == status)

        result = await self.session.execute(
            select(func.count())
            .select_from(Campaign)
            .where(and_(*conditions))
        )
        return result.scalar_one()

    async def get_with_contacts(self, id: str) -> Optional[Campaign]:
        """Get campaign with contacts."""
        result = await self.session.execute(
            select(Campaign)
            .options(selectinload(Campaign.contacts))
            .where(Campaign.id == id)
        )
        return result.scalar_one_or_none()

    async def update_status(
        self,
        campaign_id: str,
        status: str,
    ) -> Optional[Campaign]:
        """Update campaign status."""
        campaign = await self.get_by_id(campaign_id)
        if not campaign:
            return None

        campaign.status = status

        if status == "running" and not campaign.started_at:
            campaign.started_at = datetime.utcnow()
        elif status == "paused":
            campaign.paused_at = datetime.utcnow()
        elif status in ["completed", "canceled"]:
            campaign.completed_at = datetime.utcnow()

        await self.session.flush()
        await self.session.refresh(campaign)
        return campaign

    async def add_contact(
        self,
        campaign_id: str,
        organization_id: str,
        phone_number: str,
        first_name: str = None,
        last_name: str = None,
        email: str = None,
        context: Dict = None,
        extra_data: Dict = None,
    ) -> CampaignContact:
        """Add a contact to campaign."""
        contact = CampaignContact(
            campaign_id=campaign_id,
            organization_id=organization_id,
            phone_number=phone_number,
            first_name=first_name,
            last_name=last_name,
            email=email,
            context=context,
            extra_data=extra_data,
            status="pending",
        )
        self.session.add(contact)

        # Update contact count
        await self.session.execute(
            update(Campaign)
            .where(Campaign.id == campaign_id)
            .values(
                total_contacts=Campaign.total_contacts + 1,
                calls_pending=Campaign.calls_pending + 1,
            )
        )

        await self.session.flush()
        await self.session.refresh(contact)
        return contact

    async def add_contacts_bulk(
        self,
        campaign_id: str,
        organization_id: str,
        contacts: List[Dict],
    ) -> int:
        """Add multiple contacts to campaign."""
        added = 0
        for contact_data in contacts:
            try:
                contact = CampaignContact(
                    campaign_id=campaign_id,
                    organization_id=organization_id,
                    phone_number=contact_data["phone_number"],
                    first_name=contact_data.get("first_name"),
                    last_name=contact_data.get("last_name"),
                    email=contact_data.get("email"),
                    context=contact_data.get("context"),
                    extra_data=contact_data.get("extra_data"),
                    status="pending",
                )
                self.session.add(contact)
                added += 1
            except Exception:
                continue

        if added > 0:
            await self.session.execute(
                update(Campaign)
                .where(Campaign.id == campaign_id)
                .values(
                    total_contacts=Campaign.total_contacts + added,
                    calls_pending=Campaign.calls_pending + added,
                )
            )
            await self.session.flush()

        return added

    async def list_contacts(
        self,
        campaign_id: str,
        status: str = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[CampaignContact]:
        """List contacts in campaign."""
        conditions = [CampaignContact.campaign_id == campaign_id]
        if status:
            conditions.append(CampaignContact.status == status)

        result = await self.session.execute(
            select(CampaignContact)
            .where(and_(*conditions))
            .order_by(CampaignContact.created_at)
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count_contacts(
        self,
        campaign_id: str,
        status: str = None,
    ) -> int:
        """Count contacts in campaign."""
        conditions = [CampaignContact.campaign_id == campaign_id]
        if status:
            conditions.append(CampaignContact.status == status)

        result = await self.session.execute(
            select(func.count())
            .select_from(CampaignContact)
            .where(and_(*conditions))
        )
        return result.scalar_one()

    async def get_next_contact(self, campaign_id: str) -> Optional[CampaignContact]:
        """Get next contact to call."""
        result = await self.session.execute(
            select(CampaignContact)
            .where(
                and_(
                    CampaignContact.campaign_id == campaign_id,
                    CampaignContact.status == "pending",
                    or_(
                        CampaignContact.next_attempt_at == None,
                        CampaignContact.next_attempt_at <= datetime.utcnow(),
                    ),
                )
            )
            .order_by(CampaignContact.created_at)
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def update_contact_status(
        self,
        contact_id: str,
        status: str,
        call_id: str = None,
        call_outcome: str = None,
        call_duration_seconds: float = None,
        call_cost: float = None,
        next_attempt_at: datetime = None,
        notes: str = None,
    ) -> Optional[CampaignContact]:
        """Update contact call status."""
        result = await self.session.execute(
            select(CampaignContact).where(CampaignContact.id == contact_id)
        )
        contact = result.scalar_one_or_none()
        if not contact:
            return None

        old_status = contact.status
        contact.status = status
        contact.attempt_count += 1
        contact.last_attempt_at = datetime.utcnow()

        if call_id:
            contact.call_id = call_id
        if call_outcome:
            contact.call_outcome = call_outcome
        if call_duration_seconds is not None:
            contact.call_duration_seconds = call_duration_seconds
        if call_cost is not None:
            contact.call_cost = call_cost
        if next_attempt_at:
            contact.next_attempt_at = next_attempt_at
        if notes:
            contact.notes = notes

        # Update campaign stats
        campaign = await self.get_by_id(contact.campaign_id)
        if campaign:
            if old_status == "pending" or old_status == "queued":
                campaign.calls_pending = max(0, campaign.calls_pending - 1)

            if status == "completed":
                campaign.calls_completed += 1
                if call_outcome == "answered":
                    campaign.calls_successful += 1
            elif status == "failed":
                campaign.calls_failed += 1
            elif status == "calling":
                campaign.calls_in_progress += 1

            if call_duration_seconds:
                campaign.total_minutes += call_duration_seconds / 60.0
            if call_cost:
                campaign.total_cost += call_cost

        await self.session.flush()
        await self.session.refresh(contact)
        return contact

    async def get_stats(self, campaign_id: str) -> Dict[str, Any]:
        """Get campaign statistics."""
        campaign = await self.get_by_id(campaign_id)
        if not campaign:
            return {}

        total = campaign.total_contacts or 1
        completed = campaign.calls_completed or 0
        successful = campaign.calls_successful or 0

        return {
            "total_contacts": campaign.total_contacts,
            "calls_completed": completed,
            "calls_successful": successful,
            "calls_failed": campaign.calls_failed,
            "calls_pending": campaign.calls_pending,
            "calls_in_progress": campaign.calls_in_progress,
            "completion_rate": round(completed / total * 100, 2),
            "success_rate": round(successful / max(completed, 1) * 100, 2),
            "total_minutes": round(campaign.total_minutes, 2),
            "total_cost": round(campaign.total_cost, 2),
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
    "PhoneNumberRepository",
    "WebhookRepository",
    "KnowledgeBaseRepository",
    "CampaignRepository",
]
