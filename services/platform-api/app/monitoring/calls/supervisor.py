"""Call supervision service for monitoring and intervening in calls."""

from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import logging

from app.monitoring.calls.tracker import LiveCallTracker, CallInfo, get_live_call_tracker

logger = logging.getLogger(__name__)


class SupervisionMode(str, Enum):
    """Supervision modes."""
    LISTEN = "listen"  # Supervisor can only listen
    WHISPER = "whisper"  # Supervisor can talk to agent only
    BARGE = "barge"  # Supervisor can talk to all parties
    COACH = "coach"  # AI-assisted coaching suggestions


class SupervisionAction(str, Enum):
    """Actions a supervisor can take."""
    START = "start"
    STOP = "stop"
    MUTE = "mute"
    UNMUTE = "unmute"
    CHANGE_MODE = "change_mode"
    SEND_MESSAGE = "send_message"
    TAKE_OVER = "take_over"
    TRANSFER = "transfer"
    END_CALL = "end_call"


@dataclass
class SupervisorSession:
    """A supervision session."""
    session_id: str
    supervisor_id: str
    call_id: str
    mode: SupervisionMode
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    is_muted: bool = True  # Supervisors start muted
    notes: List[str] = field(default_factory=list)
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Check if session is still active."""
        return self.ended_at is None

    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        end = self.ended_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "supervisor_id": self.supervisor_id,
            "call_id": self.call_id,
            "mode": self.mode.value,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "is_muted": self.is_muted,
            "is_active": self.is_active,
            "duration_seconds": self.duration_seconds,
            "notes": self.notes,
            "actions_taken": self.actions_taken,
            "metadata": self.metadata,
        }


@dataclass
class CoachingSuggestion:
    """AI-generated coaching suggestion."""
    suggestion_id: str
    session_id: str
    suggestion_type: str  # "tone", "content", "timing", "technique"
    content: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    is_dismissed: bool = False
    is_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "suggestion_id": self.suggestion_id,
            "session_id": self.session_id,
            "suggestion_type": self.suggestion_type,
            "content": self.content,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "is_dismissed": self.is_dismissed,
            "is_applied": self.is_applied,
        }


class CallSupervisor:
    """
    Manages call supervision and intervention.

    Usage:
        supervisor = CallSupervisor(tracker)

        # Start supervision
        session = await supervisor.start_supervision(
            supervisor_id="sup_1",
            call_id="call_123",
            mode=SupervisionMode.LISTEN,
        )

        # Change mode
        await supervisor.change_mode(session.session_id, SupervisionMode.WHISPER)

        # Send whisper to agent
        await supervisor.whisper(session.session_id, "Try offering a discount")

        # Barge into call
        await supervisor.barge(session.session_id)

        # End supervision
        await supervisor.end_supervision(session.session_id)
    """

    def __init__(self, tracker: Optional[LiveCallTracker] = None):
        self.tracker = tracker or get_live_call_tracker()
        self._sessions: Dict[str, SupervisorSession] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        self._lock = asyncio.Lock()
        self._audio_streams: Dict[str, Any] = {}  # Session ID -> audio stream
        self._coaching_suggestions: Dict[str, List[CoachingSuggestion]] = {}

    async def start_supervision(
        self,
        supervisor_id: str,
        call_id: str,
        mode: SupervisionMode = SupervisionMode.LISTEN,
        **metadata,
    ) -> SupervisorSession:
        """Start supervising a call."""
        # Verify call exists and is active
        call_info = await self.tracker.get_call(call_id)
        if not call_info:
            raise ValueError(f"Call {call_id} not found")

        if not call_info.is_active:
            raise ValueError(f"Call {call_id} is not active")

        # Check if already being supervised by this supervisor
        for session in self._sessions.values():
            if session.call_id == call_id and session.supervisor_id == supervisor_id:
                if session.is_active:
                    raise ValueError(f"Already supervising call {call_id}")

        # Create session
        import uuid
        session_id = f"sup_{uuid.uuid4().hex[:12]}"

        session = SupervisorSession(
            session_id=session_id,
            supervisor_id=supervisor_id,
            call_id=call_id,
            mode=mode,
            metadata=metadata,
        )

        async with self._lock:
            self._sessions[session_id] = session
            self._coaching_suggestions[session_id] = []

        # Update call info
        await self.tracker.set_supervised(call_id, supervisor_id)

        # Record action
        await self._record_action(session_id, SupervisionAction.START, {
            "mode": mode.value,
        })

        # Set up audio stream for supervisor
        await self._setup_audio_stream(session)

        logger.info(f"Supervision started: {session_id} on call {call_id} by {supervisor_id}")

        await self._notify("supervision_started", session)

        return session

    async def end_supervision(
        self,
        session_id: str,
        reason: str = "manual",
    ) -> SupervisorSession:
        """End a supervision session."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")

            session.ended_at = datetime.utcnow()

        # Record action
        await self._record_action(session_id, SupervisionAction.STOP, {
            "reason": reason,
        })

        # Remove supervision from call
        await self.tracker.remove_supervision(session.call_id)

        # Clean up audio stream
        await self._cleanup_audio_stream(session_id)

        logger.info(f"Supervision ended: {session_id}")

        await self._notify("supervision_ended", session)

        return session

    async def change_mode(
        self,
        session_id: str,
        new_mode: SupervisionMode,
    ) -> SupervisorSession:
        """Change supervision mode."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")

            if not session.is_active:
                raise ValueError(f"Session {session_id} is not active")

            old_mode = session.mode
            session.mode = new_mode

        # Record action
        await self._record_action(session_id, SupervisionAction.CHANGE_MODE, {
            "old_mode": old_mode.value,
            "new_mode": new_mode.value,
        })

        # Update audio routing based on mode
        await self._update_audio_routing(session)

        logger.info(f"Supervision mode changed: {session_id} from {old_mode.value} to {new_mode.value}")

        await self._notify("mode_changed", session)

        return session

    async def mute(self, session_id: str) -> SupervisorSession:
        """Mute the supervisor."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")

            session.is_muted = True

        await self._record_action(session_id, SupervisionAction.MUTE, {})

        return session

    async def unmute(self, session_id: str) -> SupervisorSession:
        """Unmute the supervisor."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")

            session.is_muted = False

        await self._record_action(session_id, SupervisionAction.UNMUTE, {})

        return session

    async def whisper(
        self,
        session_id: str,
        message: str,
    ) -> None:
        """
        Send a whisper message to the agent.

        The agent can hear this, but the customer cannot.
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")

            if not session.is_active:
                raise ValueError(f"Session {session_id} is not active")

        # Change to whisper mode if not already
        if session.mode != SupervisionMode.WHISPER:
            await self.change_mode(session_id, SupervisionMode.WHISPER)

        # Record action
        await self._record_action(session_id, SupervisionAction.SEND_MESSAGE, {
            "type": "whisper",
            "message": message,
        })

        # Send whisper through audio system
        await self._send_whisper(session, message)

        await self._notify("whisper_sent", {"session": session, "message": message})

    async def barge(
        self,
        session_id: str,
        announcement: Optional[str] = None,
    ) -> None:
        """
        Barge into the call.

        The supervisor's voice is heard by all parties.
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")

            if not session.is_active:
                raise ValueError(f"Session {session_id} is not active")

        # Change to barge mode
        await self.change_mode(session_id, SupervisionMode.BARGE)

        # Unmute
        await self.unmute(session_id)

        # Record action
        await self._record_action(session_id, SupervisionAction.SEND_MESSAGE, {
            "type": "barge",
            "announcement": announcement,
        })

        # If there's an announcement, play it
        if announcement:
            await self._play_announcement(session, announcement)

        await self._notify("barge", {"session": session, "announcement": announcement})

    async def take_over(
        self,
        session_id: str,
    ) -> None:
        """
        Take over the call from the AI agent.

        The supervisor becomes the primary handler.
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")

            if not session.is_active:
                raise ValueError(f"Session {session_id} is not active")

        # Record action
        await self._record_action(session_id, SupervisionAction.TAKE_OVER, {})

        # Update call to indicate human takeover
        call_info = await self.tracker.get_call(session.call_id)
        if call_info:
            call_info.metadata["human_takeover"] = True
            call_info.metadata["takeover_by"] = session.supervisor_id
            call_info.metadata["takeover_at"] = datetime.utcnow().isoformat()

        logger.info(f"Call {session.call_id} taken over by supervisor {session.supervisor_id}")

        await self._notify("call_taken_over", session)

    async def transfer_call(
        self,
        session_id: str,
        destination: str,
        transfer_type: str = "blind",  # "blind" or "warm"
    ) -> None:
        """Transfer the call to another destination."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")

        # Record action
        await self._record_action(session_id, SupervisionAction.TRANSFER, {
            "destination": destination,
            "transfer_type": transfer_type,
        })

        logger.info(f"Call {session.call_id} transferred to {destination}")

        await self._notify("call_transferred", {
            "session": session,
            "destination": destination,
            "transfer_type": transfer_type,
        })

    async def end_call(
        self,
        session_id: str,
        reason: str = "supervisor_ended",
    ) -> None:
        """End the call being supervised."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")

        # Record action
        await self._record_action(session_id, SupervisionAction.END_CALL, {
            "reason": reason,
        })

        # End the call
        await self.tracker.end_call(session.call_id, reason)

        # End supervision
        await self.end_supervision(session_id, "call_ended")

        logger.info(f"Call {session.call_id} ended by supervisor")

    async def add_note(
        self,
        session_id: str,
        note: str,
    ) -> SupervisorSession:
        """Add a note to the supervision session."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")

            session.notes.append(f"[{datetime.utcnow().isoformat()}] {note}")

        return session

    async def get_session(self, session_id: str) -> Optional[SupervisorSession]:
        """Get a supervision session."""
        async with self._lock:
            return self._sessions.get(session_id)

    async def get_sessions_for_call(
        self,
        call_id: str,
        active_only: bool = True,
    ) -> List[SupervisorSession]:
        """Get all supervision sessions for a call."""
        async with self._lock:
            sessions = [s for s in self._sessions.values() if s.call_id == call_id]

        if active_only:
            sessions = [s for s in sessions if s.is_active]

        return sessions

    async def get_sessions_by_supervisor(
        self,
        supervisor_id: str,
        active_only: bool = True,
    ) -> List[SupervisorSession]:
        """Get all sessions for a supervisor."""
        async with self._lock:
            sessions = [s for s in self._sessions.values() if s.supervisor_id == supervisor_id]

        if active_only:
            sessions = [s for s in sessions if s.is_active]

        return sessions

    async def add_coaching_suggestion(
        self,
        session_id: str,
        suggestion_type: str,
        content: str,
        confidence: float = 0.8,
    ) -> CoachingSuggestion:
        """Add an AI coaching suggestion."""
        import uuid

        suggestion = CoachingSuggestion(
            suggestion_id=f"sug_{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            suggestion_type=suggestion_type,
            content=content,
            confidence=confidence,
        )

        async with self._lock:
            if session_id not in self._coaching_suggestions:
                self._coaching_suggestions[session_id] = []
            self._coaching_suggestions[session_id].append(suggestion)

        await self._notify("coaching_suggestion", suggestion)

        return suggestion

    async def get_coaching_suggestions(
        self,
        session_id: str,
    ) -> List[CoachingSuggestion]:
        """Get coaching suggestions for a session."""
        async with self._lock:
            return self._coaching_suggestions.get(session_id, [])

    async def dismiss_suggestion(
        self,
        suggestion_id: str,
    ) -> None:
        """Dismiss a coaching suggestion."""
        async with self._lock:
            for suggestions in self._coaching_suggestions.values():
                for suggestion in suggestions:
                    if suggestion.suggestion_id == suggestion_id:
                        suggestion.is_dismissed = True
                        return

    async def apply_suggestion(
        self,
        suggestion_id: str,
    ) -> None:
        """Mark a coaching suggestion as applied."""
        async with self._lock:
            for suggestions in self._coaching_suggestions.values():
                for suggestion in suggestions:
                    if suggestion.suggestion_id == suggestion_id:
                        suggestion.is_applied = True
                        return

    def on_event(
        self,
        event_type: str,
        callback: Callable[[Any], Awaitable[None]],
    ) -> None:
        """Register callback for supervision events."""
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)

    async def _record_action(
        self,
        session_id: str,
        action: SupervisionAction,
        data: Dict[str, Any],
    ) -> None:
        """Record an action taken during supervision."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.actions_taken.append({
                    "action": action.value,
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": data,
                })

    async def _notify(self, event_type: str, data: Any) -> None:
        """Notify callbacks of an event."""
        callbacks = self._callbacks.get(event_type, [])
        for callback in callbacks:
            try:
                await callback(data)
            except Exception as e:
                logger.error(f"Supervision callback error: {e}")

    async def _setup_audio_stream(self, session: SupervisorSession) -> None:
        """Set up audio stream for supervisor."""
        # In a real implementation, this would set up WebRTC or similar
        self._audio_streams[session.session_id] = {
            "call_id": session.call_id,
            "mode": session.mode.value,
            "created_at": datetime.utcnow(),
        }

    async def _cleanup_audio_stream(self, session_id: str) -> None:
        """Clean up audio stream."""
        if session_id in self._audio_streams:
            del self._audio_streams[session_id]

    async def _update_audio_routing(self, session: SupervisorSession) -> None:
        """Update audio routing based on supervision mode."""
        if session.session_id in self._audio_streams:
            self._audio_streams[session.session_id]["mode"] = session.mode.value

    async def _send_whisper(self, session: SupervisorSession, message: str) -> None:
        """Send whisper message to agent."""
        # In a real implementation, this would send audio to agent only
        logger.debug(f"Whisper to agent on call {session.call_id}: {message}")

    async def _play_announcement(self, session: SupervisorSession, announcement: str) -> None:
        """Play announcement to all parties."""
        # In a real implementation, this would play TTS audio
        logger.debug(f"Announcement on call {session.call_id}: {announcement}")
