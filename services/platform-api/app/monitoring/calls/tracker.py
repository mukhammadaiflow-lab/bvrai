"""Live call tracking service."""

from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class CallState(str, Enum):
    """Call states."""
    INITIATING = "initiating"
    RINGING = "ringing"
    ANSWERED = "answered"
    ON_HOLD = "on_hold"
    TRANSFERRING = "transferring"
    ENDING = "ending"
    COMPLETED = "completed"
    FAILED = "failed"


class CallDirection(str, Enum):
    """Call direction."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"


@dataclass
class Participant:
    """A call participant."""
    id: str
    type: str  # "user", "agent", "external"
    name: Optional[str] = None
    phone_number: Optional[str] = None
    joined_at: datetime = field(default_factory=datetime.utcnow)
    is_muted: bool = False
    is_on_hold: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "phone_number": self.phone_number,
            "joined_at": self.joined_at.isoformat(),
            "is_muted": self.is_muted,
            "is_on_hold": self.is_on_hold,
        }


@dataclass
class CallInfo:
    """Information about an active call."""
    call_id: str
    account_id: str
    agent_id: str
    state: CallState
    direction: CallDirection
    started_at: datetime = field(default_factory=datetime.utcnow)
    answered_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    caller_number: Optional[str] = None
    called_number: Optional[str] = None
    participants: List[Participant] = field(default_factory=list)
    current_intent: Optional[str] = None
    sentiment_score: float = 0.0
    is_supervised: bool = False
    supervisor_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Get call duration in seconds."""
        end = self.ended_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()

    @property
    def talk_time_seconds(self) -> float:
        """Get talk time (from answer) in seconds."""
        if not self.answered_at:
            return 0.0
        end = self.ended_at or datetime.utcnow()
        return (end - self.answered_at).total_seconds()

    @property
    def is_active(self) -> bool:
        """Check if call is still active."""
        return self.state not in [CallState.COMPLETED, CallState.FAILED]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "call_id": self.call_id,
            "account_id": self.account_id,
            "agent_id": self.agent_id,
            "state": self.state.value,
            "direction": self.direction.value,
            "started_at": self.started_at.isoformat(),
            "answered_at": self.answered_at.isoformat() if self.answered_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "caller_number": self.caller_number,
            "called_number": self.called_number,
            "participants": [p.to_dict() for p in self.participants],
            "duration_seconds": self.duration_seconds,
            "talk_time_seconds": self.talk_time_seconds,
            "current_intent": self.current_intent,
            "sentiment_score": self.sentiment_score,
            "is_supervised": self.is_supervised,
            "supervisor_id": self.supervisor_id,
            "is_active": self.is_active,
            "metadata": self.metadata,
        }


@dataclass
class CallEvent:
    """An event that occurred during a call."""
    event_type: str
    call_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type,
            "call_id": self.call_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }


class LiveCallTracker:
    """
    Tracks all active calls in real-time.

    Usage:
        tracker = LiveCallTracker()

        # Register call
        call_info = await tracker.register_call(
            call_id="call_123",
            account_id="acc_1",
            agent_id="agent_1",
            direction=CallDirection.INBOUND,
        )

        # Update call state
        await tracker.update_state("call_123", CallState.ANSWERED)

        # Get active calls
        calls = await tracker.get_active_calls()

        # Subscribe to updates
        tracker.on_call_event(handle_event)
    """

    def __init__(self):
        self._calls: Dict[str, CallInfo] = {}
        self._events: Dict[str, List[CallEvent]] = {}
        self._callbacks: List[Callable[[CallEvent], Awaitable[None]]] = []
        self._lock = asyncio.Lock()

    async def register_call(
        self,
        call_id: str,
        account_id: str,
        agent_id: str,
        direction: CallDirection,
        caller_number: Optional[str] = None,
        called_number: Optional[str] = None,
        **metadata,
    ) -> CallInfo:
        """Register a new call."""
        call_info = CallInfo(
            call_id=call_id,
            account_id=account_id,
            agent_id=agent_id,
            state=CallState.INITIATING,
            direction=direction,
            caller_number=caller_number,
            called_number=called_number,
            metadata=metadata,
        )

        async with self._lock:
            self._calls[call_id] = call_info
            self._events[call_id] = []

        await self._emit_event(CallEvent(
            event_type="call.registered",
            call_id=call_id,
            data={"direction": direction.value},
        ))

        logger.info(f"Call registered: {call_id}")
        return call_info

    async def update_state(
        self,
        call_id: str,
        state: CallState,
        **data,
    ) -> Optional[CallInfo]:
        """Update call state."""
        async with self._lock:
            call_info = self._calls.get(call_id)
            if not call_info:
                return None

            old_state = call_info.state
            call_info.state = state

            # Update timestamps
            if state == CallState.ANSWERED and not call_info.answered_at:
                call_info.answered_at = datetime.utcnow()
            elif state in [CallState.COMPLETED, CallState.FAILED]:
                call_info.ended_at = datetime.utcnow()

            # Update metadata
            call_info.metadata.update(data)

        await self._emit_event(CallEvent(
            event_type="call.state_changed",
            call_id=call_id,
            data={"old_state": old_state.value, "new_state": state.value, **data},
        ))

        logger.info(f"Call {call_id} state changed: {old_state.value} -> {state.value}")
        return call_info

    async def add_participant(
        self,
        call_id: str,
        participant: Participant,
    ) -> Optional[CallInfo]:
        """Add a participant to the call."""
        async with self._lock:
            call_info = self._calls.get(call_id)
            if not call_info:
                return None

            call_info.participants.append(participant)

        await self._emit_event(CallEvent(
            event_type="call.participant_joined",
            call_id=call_id,
            data=participant.to_dict(),
        ))

        return call_info

    async def remove_participant(
        self,
        call_id: str,
        participant_id: str,
    ) -> Optional[CallInfo]:
        """Remove a participant from the call."""
        async with self._lock:
            call_info = self._calls.get(call_id)
            if not call_info:
                return None

            call_info.participants = [
                p for p in call_info.participants
                if p.id != participant_id
            ]

        await self._emit_event(CallEvent(
            event_type="call.participant_left",
            call_id=call_id,
            data={"participant_id": participant_id},
        ))

        return call_info

    async def update_intent(
        self,
        call_id: str,
        intent: str,
    ) -> Optional[CallInfo]:
        """Update the detected intent for a call."""
        async with self._lock:
            call_info = self._calls.get(call_id)
            if not call_info:
                return None

            call_info.current_intent = intent

        await self._emit_event(CallEvent(
            event_type="call.intent_detected",
            call_id=call_id,
            data={"intent": intent},
        ))

        return call_info

    async def update_sentiment(
        self,
        call_id: str,
        sentiment_score: float,
    ) -> Optional[CallInfo]:
        """Update the sentiment score for a call."""
        async with self._lock:
            call_info = self._calls.get(call_id)
            if not call_info:
                return None

            call_info.sentiment_score = sentiment_score

        await self._emit_event(CallEvent(
            event_type="call.sentiment_updated",
            call_id=call_id,
            data={"sentiment_score": sentiment_score},
        ))

        return call_info

    async def set_supervised(
        self,
        call_id: str,
        supervisor_id: str,
    ) -> Optional[CallInfo]:
        """Mark a call as being supervised."""
        async with self._lock:
            call_info = self._calls.get(call_id)
            if not call_info:
                return None

            call_info.is_supervised = True
            call_info.supervisor_id = supervisor_id

        await self._emit_event(CallEvent(
            event_type="call.supervision_started",
            call_id=call_id,
            data={"supervisor_id": supervisor_id},
        ))

        return call_info

    async def remove_supervision(self, call_id: str) -> Optional[CallInfo]:
        """Remove supervision from a call."""
        async with self._lock:
            call_info = self._calls.get(call_id)
            if not call_info:
                return None

            supervisor_id = call_info.supervisor_id
            call_info.is_supervised = False
            call_info.supervisor_id = None

        await self._emit_event(CallEvent(
            event_type="call.supervision_ended",
            call_id=call_id,
            data={"supervisor_id": supervisor_id},
        ))

        return call_info

    async def end_call(
        self,
        call_id: str,
        reason: str = "normal",
    ) -> Optional[CallInfo]:
        """End a call."""
        state = CallState.COMPLETED if reason == "normal" else CallState.FAILED
        return await self.update_state(call_id, state, end_reason=reason)

    async def get_call(self, call_id: str) -> Optional[CallInfo]:
        """Get a specific call."""
        async with self._lock:
            return self._calls.get(call_id)

    async def get_active_calls(
        self,
        account_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[CallInfo]:
        """Get all active calls, optionally filtered."""
        async with self._lock:
            calls = [c for c in self._calls.values() if c.is_active]

        if account_id:
            calls = [c for c in calls if c.account_id == account_id]

        if agent_id:
            calls = [c for c in calls if c.agent_id == agent_id]

        return calls

    async def get_all_calls(
        self,
        include_completed: bool = False,
    ) -> List[CallInfo]:
        """Get all tracked calls."""
        async with self._lock:
            if include_completed:
                return list(self._calls.values())
            return [c for c in self._calls.values() if c.is_active]

    async def get_call_events(
        self,
        call_id: str,
        event_type: Optional[str] = None,
    ) -> List[CallEvent]:
        """Get events for a specific call."""
        async with self._lock:
            events = self._events.get(call_id, [])

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events

    async def get_active_call_count(
        self,
        account_id: Optional[str] = None,
    ) -> int:
        """Get count of active calls."""
        calls = await self.get_active_calls(account_id=account_id)
        return len(calls)

    async def cleanup_completed(
        self,
        max_age_seconds: int = 3600,
    ) -> int:
        """Remove completed calls older than max_age."""
        cutoff = datetime.utcnow() - timedelta(seconds=max_age_seconds)
        removed = 0

        async with self._lock:
            to_remove = [
                call_id for call_id, call_info in self._calls.items()
                if not call_info.is_active and (
                    call_info.ended_at and call_info.ended_at < cutoff
                )
            ]

            for call_id in to_remove:
                del self._calls[call_id]
                if call_id in self._events:
                    del self._events[call_id]
                removed += 1

        if removed > 0:
            logger.info(f"Cleaned up {removed} completed calls")

        return removed

    def on_call_event(
        self,
        callback: Callable[[CallEvent], Awaitable[None]],
    ) -> None:
        """Register callback for call events."""
        self._callbacks.append(callback)

    async def _emit_event(self, event: CallEvent) -> None:
        """Emit a call event."""
        async with self._lock:
            if event.call_id in self._events:
                self._events[event.call_id].append(event)

        for callback in self._callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Call event callback error: {e}")


class CallStatsAggregator:
    """
    Aggregates call statistics in real-time.

    Usage:
        aggregator = CallStatsAggregator(tracker)

        # Get current stats
        stats = await aggregator.get_stats()

        # Get agent stats
        agent_stats = await aggregator.get_agent_stats("agent_1")
    """

    def __init__(self, tracker: LiveCallTracker):
        self.tracker = tracker
        self._stats_cache: Dict[str, Any] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(seconds=5)

    async def get_stats(
        self,
        account_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get aggregated call statistics."""
        calls = await self.tracker.get_all_calls(include_completed=True)

        if account_id:
            calls = [c for c in calls if c.account_id == account_id]

        active_calls = [c for c in calls if c.is_active]
        completed_calls = [c for c in calls if not c.is_active]

        # Calculate stats
        total_duration = sum(c.duration_seconds for c in completed_calls)
        total_talk_time = sum(c.talk_time_seconds for c in completed_calls)
        avg_duration = total_duration / len(completed_calls) if completed_calls else 0
        avg_talk_time = total_talk_time / len(completed_calls) if completed_calls else 0

        # Count by state
        state_counts = {}
        for call in calls:
            state = call.state.value
            state_counts[state] = state_counts.get(state, 0) + 1

        # Direction counts
        inbound = len([c for c in calls if c.direction == CallDirection.INBOUND])
        outbound = len([c for c in calls if c.direction == CallDirection.OUTBOUND])

        # Sentiment distribution
        positive = len([c for c in calls if c.sentiment_score > 0.3])
        negative = len([c for c in calls if c.sentiment_score < -0.3])
        neutral = len(calls) - positive - negative

        return {
            "total_calls": len(calls),
            "active_calls": len(active_calls),
            "completed_calls": len(completed_calls),
            "state_counts": state_counts,
            "direction": {
                "inbound": inbound,
                "outbound": outbound,
            },
            "duration": {
                "total_seconds": total_duration,
                "average_seconds": avg_duration,
                "total_talk_seconds": total_talk_time,
                "average_talk_seconds": avg_talk_time,
            },
            "sentiment": {
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
            },
            "supervised_count": len([c for c in active_calls if c.is_supervised]),
        }

    async def get_agent_stats(
        self,
        agent_id: str,
    ) -> Dict[str, Any]:
        """Get statistics for a specific agent."""
        calls = await self.tracker.get_all_calls(include_completed=True)
        calls = [c for c in calls if c.agent_id == agent_id]

        if not calls:
            return {
                "agent_id": agent_id,
                "total_calls": 0,
                "active_calls": 0,
                "is_available": True,
            }

        active = [c for c in calls if c.is_active]
        completed = [c for c in calls if not c.is_active]

        total_talk_time = sum(c.talk_time_seconds for c in completed)
        avg_talk_time = total_talk_time / len(completed) if completed else 0

        return {
            "agent_id": agent_id,
            "total_calls": len(calls),
            "active_calls": len(active),
            "completed_calls": len(completed),
            "is_available": len(active) == 0,
            "current_call_id": active[0].call_id if active else None,
            "total_talk_time_seconds": total_talk_time,
            "average_talk_time_seconds": avg_talk_time,
            "avg_sentiment": sum(c.sentiment_score for c in calls) / len(calls) if calls else 0,
        }


# Global tracker
_live_call_tracker: Optional[LiveCallTracker] = None


def get_live_call_tracker() -> LiveCallTracker:
    """Get or create the global live call tracker."""
    global _live_call_tracker
    if _live_call_tracker is None:
        _live_call_tracker = LiveCallTracker()
    return _live_call_tracker
