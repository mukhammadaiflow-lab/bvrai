"""Agent service for CRUD operations."""

from typing import Optional
from uuid import UUID

import structlog
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Agent, AgentStatus, Call, CallStatus
from app.agents.schemas import AgentCreate, AgentUpdate, AgentStats

logger = structlog.get_logger()


class AgentService:
    """Service for agent operations."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.logger = logger.bind(service="agent")

    async def create(self, owner_id: UUID, data: AgentCreate) -> Agent:
        """Create a new agent."""
        agent = Agent(
            owner_id=owner_id,
            name=data.name,
            description=data.description,
            status=AgentStatus.DRAFT,
            voice_id=data.voice_id,
            voice_name=data.voice_name,
            language=data.language,
            system_prompt=data.system_prompt,
            greeting_message=data.greeting_message,
            fallback_message=data.fallback_message,
            goodbye_message=data.goodbye_message,
            llm_provider=data.llm_provider,
            llm_model=data.llm_model,
            temperature=data.temperature,
            max_tokens=data.max_tokens,
            phone_number=data.phone_number,
            forward_number=data.forward_number,
            tools=[t.model_dump() for t in data.tools],
            settings=data.settings,
        )

        self.db.add(agent)
        await self.db.flush()
        await self.db.refresh(agent)

        self.logger.info("Agent created", agent_id=str(agent.id), name=agent.name)
        return agent

    async def get(self, agent_id: UUID, owner_id: Optional[UUID] = None) -> Optional[Agent]:
        """Get an agent by ID."""
        query = select(Agent).where(Agent.id == agent_id)

        if owner_id:
            query = query.where(Agent.owner_id == owner_id)

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def list(
        self,
        owner_id: UUID,
        status: Optional[AgentStatus] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Agent], int]:
        """List agents for an owner."""
        query = select(Agent).where(Agent.owner_id == owner_id)

        if status:
            query = query.where(Agent.status == status)

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total = await self.db.execute(count_query)
        total_count = total.scalar() or 0

        # Paginate
        query = query.order_by(Agent.created_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)

        result = await self.db.execute(query)
        agents = result.scalars().all()

        return list(agents), total_count

    async def update(
        self,
        agent_id: UUID,
        data: AgentUpdate,
        owner_id: Optional[UUID] = None,
    ) -> Optional[Agent]:
        """Update an agent."""
        agent = await self.get(agent_id, owner_id)
        if not agent:
            return None

        # Update fields
        update_data = data.model_dump(exclude_unset=True)

        # Handle tools serialization
        if "tools" in update_data and update_data["tools"]:
            update_data["tools"] = [
                t.model_dump() if hasattr(t, "model_dump") else t
                for t in update_data["tools"]
            ]

        for field, value in update_data.items():
            setattr(agent, field, value)

        await self.db.flush()
        await self.db.refresh(agent)

        self.logger.info("Agent updated", agent_id=str(agent_id))
        return agent

    async def delete(self, agent_id: UUID, owner_id: Optional[UUID] = None) -> bool:
        """Delete an agent."""
        agent = await self.get(agent_id, owner_id)
        if not agent:
            return False

        await self.db.delete(agent)
        await self.db.flush()

        self.logger.info("Agent deleted", agent_id=str(agent_id))
        return True

    async def get_stats(self, agent_id: UUID) -> AgentStats:
        """Get agent statistics."""
        # Total calls
        total_query = select(func.count()).where(Call.agent_id == agent_id)
        total_result = await self.db.execute(total_query)
        total_calls = total_result.scalar() or 0

        # Completed calls
        completed_query = select(func.count()).where(
            and_(Call.agent_id == agent_id, Call.status == CallStatus.COMPLETED)
        )
        completed_result = await self.db.execute(completed_query)
        completed_calls = completed_result.scalar() or 0

        # Failed calls
        failed_query = select(func.count()).where(
            and_(Call.agent_id == agent_id, Call.status == CallStatus.FAILED)
        )
        failed_result = await self.db.execute(failed_query)
        failed_calls = failed_result.scalar() or 0

        # Average duration
        avg_query = select(func.avg(Call.duration_seconds)).where(
            and_(
                Call.agent_id == agent_id,
                Call.status == CallStatus.COMPLETED,
            )
        )
        avg_result = await self.db.execute(avg_query)
        avg_duration = avg_result.scalar() or 0

        # Total duration
        total_dur_query = select(func.sum(Call.duration_seconds)).where(
            Call.agent_id == agent_id
        )
        total_dur_result = await self.db.execute(total_dur_query)
        total_duration = total_dur_result.scalar() or 0

        return AgentStats(
            agent_id=agent_id,
            total_calls=total_calls,
            completed_calls=completed_calls,
            failed_calls=failed_calls,
            avg_duration_seconds=float(avg_duration),
            total_duration_seconds=int(total_duration),
            calls_today=0,  # TODO: Implement time-based queries
            calls_this_week=0,
            calls_this_month=0,
        )

    async def activate(self, agent_id: UUID, owner_id: Optional[UUID] = None) -> Optional[Agent]:
        """Activate an agent."""
        return await self.update(
            agent_id,
            AgentUpdate(status=AgentStatus.ACTIVE),
            owner_id,
        )

    async def pause(self, agent_id: UUID, owner_id: Optional[UUID] = None) -> Optional[Agent]:
        """Pause an agent."""
        return await self.update(
            agent_id,
            AgentUpdate(status=AgentStatus.PAUSED),
            owner_id,
        )

    async def archive(self, agent_id: UUID, owner_id: Optional[UUID] = None) -> Optional[Agent]:
        """Archive an agent."""
        return await self.update(
            agent_id,
            AgentUpdate(status=AgentStatus.ARCHIVED),
            owner_id,
        )
