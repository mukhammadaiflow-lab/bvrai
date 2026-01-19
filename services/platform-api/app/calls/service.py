"""Call service for CRUD operations."""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID, uuid4

import structlog
from sqlalchemy import select, func, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Call, CallLog, CallStatus, CallDirection
from app.calls.schemas import CallCreate, CallSummary

logger = structlog.get_logger()


class CallService:
    """Service for call operations."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.logger = logger.bind(service="call")

    async def create(
        self,
        agent_id: UUID,
        direction: CallDirection,
        from_number: Optional[str] = None,
        to_number: Optional[str] = None,
        twilio_call_sid: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Call:
        """Create a new call record."""
        call = Call(
            agent_id=agent_id,
            session_id=str(uuid4()),
            direction=direction,
            status=CallStatus.QUEUED,
            from_number=from_number,
            to_number=to_number,
            twilio_call_sid=twilio_call_sid,
            metadata=metadata or {},
        )

        self.db.add(call)
        await self.db.flush()
        await self.db.refresh(call)

        self.logger.info(
            "Call created",
            call_id=str(call.id),
            session_id=call.session_id,
            direction=direction.value,
        )

        return call

    async def get(self, call_id: UUID) -> Optional[Call]:
        """Get a call by ID."""
        query = select(Call).where(Call.id == call_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_by_session(self, session_id: str) -> Optional[Call]:
        """Get a call by session ID."""
        query = select(Call).where(Call.session_id == session_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_by_twilio_sid(self, twilio_call_sid: str) -> Optional[Call]:
        """Get a call by Twilio Call SID."""
        query = select(Call).where(Call.twilio_call_sid == twilio_call_sid)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def list(
        self,
        agent_id: Optional[UUID] = None,
        status: Optional[CallStatus] = None,
        direction: Optional[CallDirection] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Call], int]:
        """List calls with filters."""
        query = select(Call)

        # Apply filters
        conditions = []
        if agent_id:
            conditions.append(Call.agent_id == agent_id)
        if status:
            conditions.append(Call.status == status)
        if direction:
            conditions.append(Call.direction == direction)
        if from_date:
            conditions.append(Call.created_at >= from_date)
        if to_date:
            conditions.append(Call.created_at <= to_date)

        if conditions:
            query = query.where(and_(*conditions))

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total = await self.db.execute(count_query)
        total_count = total.scalar() or 0

        # Paginate
        query = query.order_by(desc(Call.created_at))
        query = query.offset((page - 1) * page_size).limit(page_size)

        result = await self.db.execute(query)
        calls = result.scalars().all()

        return list(calls), total_count

    async def update_status(
        self,
        call_id: UUID,
        status: CallStatus,
        ended_at: Optional[datetime] = None,
    ) -> Optional[Call]:
        """Update call status."""
        call = await self.get(call_id)
        if not call:
            return None

        call.status = status

        if status == CallStatus.IN_PROGRESS and not call.started_at:
            call.started_at = datetime.utcnow()

        if ended_at:
            call.ended_at = ended_at
            if call.started_at:
                call.duration_seconds = int((ended_at - call.started_at).total_seconds())

        await self.db.flush()
        await self.db.refresh(call)

        self.logger.info(
            "Call status updated",
            call_id=str(call_id),
            status=status.value,
        )

        return call

    async def end_call(self, call_id: UUID, status: CallStatus = CallStatus.COMPLETED) -> Optional[Call]:
        """End a call."""
        return await self.update_status(call_id, status, ended_at=datetime.utcnow())

    async def add_log(
        self,
        call_id: UUID,
        event_type: str,
        speaker: Optional[str] = None,
        content: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> CallLog:
        """Add a log entry to a call."""
        log = CallLog(
            call_id=call_id,
            event_type=event_type,
            speaker=speaker,
            content=content,
            metadata=metadata or {},
        )

        self.db.add(log)
        await self.db.flush()
        await self.db.refresh(log)

        return log

    async def get_logs(self, call_id: UUID) -> list[CallLog]:
        """Get all logs for a call."""
        query = (
            select(CallLog)
            .where(CallLog.call_id == call_id)
            .order_by(CallLog.timestamp)
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_transcript(self, call_id: UUID) -> list[CallLog]:
        """Get transcript entries for a call."""
        query = (
            select(CallLog)
            .where(
                and_(
                    CallLog.call_id == call_id,
                    CallLog.event_type.in_(["user_speech", "agent_speech", "transcript"]),
                )
            )
            .order_by(CallLog.timestamp)
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_active_calls(self, agent_id: Optional[UUID] = None) -> list[Call]:
        """Get all active calls."""
        query = select(Call).where(
            Call.status.in_([CallStatus.QUEUED, CallStatus.RINGING, CallStatus.IN_PROGRESS])
        )

        if agent_id:
            query = query.where(Call.agent_id == agent_id)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_call_stats(
        self,
        agent_id: Optional[UUID] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> dict:
        """Get call statistics."""
        conditions = []
        if agent_id:
            conditions.append(Call.agent_id == agent_id)
        if from_date:
            conditions.append(Call.created_at >= from_date)
        if to_date:
            conditions.append(Call.created_at <= to_date)

        base_query = select(Call)
        if conditions:
            base_query = base_query.where(and_(*conditions))

        # Total calls
        total_query = select(func.count()).select_from(base_query.subquery())
        total_result = await self.db.execute(total_query)
        total_calls = total_result.scalar() or 0

        # Completed calls
        completed_query = select(func.count()).where(
            and_(
                Call.status == CallStatus.COMPLETED,
                *conditions if conditions else [True],
            )
        )
        completed_result = await self.db.execute(completed_query)
        completed_calls = completed_result.scalar() or 0

        # Failed calls
        failed_query = select(func.count()).where(
            and_(
                Call.status == CallStatus.FAILED,
                *conditions if conditions else [True],
            )
        )
        failed_result = await self.db.execute(failed_query)
        failed_calls = failed_result.scalar() or 0

        # Average duration
        avg_query = select(func.avg(Call.duration_seconds)).where(
            and_(
                Call.status == CallStatus.COMPLETED,
                Call.duration_seconds.isnot(None),
                *conditions if conditions else [True],
            )
        )
        avg_result = await self.db.execute(avg_query)
        avg_duration = avg_result.scalar() or 0

        # Total duration
        total_dur_query = select(func.sum(Call.duration_seconds)).where(
            and_(
                Call.duration_seconds.isnot(None),
                *conditions if conditions else [True],
            )
        )
        total_dur_result = await self.db.execute(total_dur_query)
        total_duration = total_dur_result.scalar() or 0

        # Inbound vs outbound
        inbound_query = select(func.count()).where(
            and_(
                Call.direction == CallDirection.INBOUND,
                *conditions if conditions else [True],
            )
        )
        inbound_result = await self.db.execute(inbound_query)
        inbound_calls = inbound_result.scalar() or 0

        return {
            "total_calls": total_calls,
            "completed_calls": completed_calls,
            "failed_calls": failed_calls,
            "inbound_calls": inbound_calls,
            "outbound_calls": total_calls - inbound_calls,
            "avg_duration_seconds": float(avg_duration),
            "total_duration_seconds": int(total_duration),
            "completion_rate": completed_calls / total_calls if total_calls > 0 else 0,
        }

    async def generate_summary(self, call_id: UUID) -> Optional[CallSummary]:
        """Generate AI summary for a call (placeholder)."""
        call = await self.get(call_id)
        if not call:
            return None

        transcript = await self.get_transcript(call_id)

        # TODO: Integrate with AI Orchestrator for real summary
        # For now, return a placeholder

        return CallSummary(
            call_id=call_id,
            summary="Call summary will be generated by AI.",
            key_points=["Point 1", "Point 2"],
            action_items=[],
            sentiment="neutral",
            topics=["general"],
        )
