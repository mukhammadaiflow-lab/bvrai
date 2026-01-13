"""
Call Session Management Module

This module provides comprehensive session management for voice calls,
including lifecycle management, state persistence, and session pooling.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

from .base import (
    CallSession,
    CallState,
    CallDirection,
    CallConfig,
    CallMetrics,
    EventType,
    OrchestratorEvent,
    TranscriptEntry,
    ConversationTurn,
    ParticipantRole,
)


logger = logging.getLogger(__name__)

T = TypeVar("T")


class SessionStorageBackend(ABC):
    """Abstract base class for session storage backends."""

    @abstractmethod
    async def save(self, session: CallSession) -> None:
        """Save session to storage."""
        pass

    @abstractmethod
    async def load(self, session_id: str) -> Optional[CallSession]:
        """Load session from storage."""
        pass

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """Delete session from storage."""
        pass

    @abstractmethod
    async def list_active(
        self,
        organization_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[CallSession]:
        """List active sessions."""
        pass

    @abstractmethod
    async def cleanup_expired(self, max_age_seconds: int) -> int:
        """Clean up expired sessions."""
        pass


class InMemorySessionStorage(SessionStorageBackend):
    """In-memory session storage for development and testing."""

    def __init__(self):
        """Initialize in-memory storage."""
        self._sessions: Dict[str, CallSession] = {}
        self._lock = asyncio.Lock()

    async def save(self, session: CallSession) -> None:
        """Save session to memory."""
        async with self._lock:
            self._sessions[session.id] = session

    async def load(self, session_id: str) -> Optional[CallSession]:
        """Load session from memory."""
        return self._sessions.get(session_id)

    async def delete(self, session_id: str) -> None:
        """Delete session from memory."""
        async with self._lock:
            self._sessions.pop(session_id, None)

    async def list_active(
        self,
        organization_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[CallSession]:
        """List active sessions."""
        sessions = []

        for session in self._sessions.values():
            # Filter by organization if specified
            if organization_id and session.organization_id != organization_id:
                continue

            # Only include active sessions
            if session.state not in (
                CallState.COMPLETED,
                CallState.FAILED,
                CallState.CANCELED,
            ):
                sessions.append(session)

            if len(sessions) >= limit:
                break

        return sessions

    async def cleanup_expired(self, max_age_seconds: int) -> int:
        """Clean up expired sessions."""
        now = datetime.utcnow()
        expired = []

        async with self._lock:
            for session_id, session in self._sessions.items():
                age = (now - session.started_at).total_seconds()
                if age > max_age_seconds:
                    expired.append(session_id)

            for session_id in expired:
                del self._sessions[session_id]

        return len(expired)


class RedisSessionStorage(SessionStorageBackend):
    """Redis-based session storage for production use."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "call_session:",
        ttl_seconds: int = 86400,  # 24 hours
    ):
        """
        Initialize Redis storage.

        Args:
            redis_url: Redis connection URL
            key_prefix: Key prefix for sessions
            ttl_seconds: TTL for session keys
        """
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds
        self._redis: Optional[Any] = None

    async def _get_redis(self) -> Any:
        """Get Redis connection (lazy initialization)."""
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._redis = await redis.from_url(self.redis_url)
            except ImportError:
                logger.warning("redis package not installed, using mock")
                return None
        return self._redis

    async def save(self, session: CallSession) -> None:
        """Save session to Redis."""
        redis = await self._get_redis()
        if redis is None:
            return

        key = f"{self.key_prefix}{session.id}"
        data = session.to_dict()

        import json
        await redis.setex(key, self.ttl_seconds, json.dumps(data))

        # Add to organization index
        org_key = f"{self.key_prefix}org:{session.organization_id}"
        await redis.sadd(org_key, session.id)
        await redis.expire(org_key, self.ttl_seconds)

    async def load(self, session_id: str) -> Optional[CallSession]:
        """Load session from Redis."""
        redis = await self._get_redis()
        if redis is None:
            return None

        key = f"{self.key_prefix}{session_id}"
        data = await redis.get(key)

        if data is None:
            return None

        import json
        session_dict = json.loads(data)
        return CallSession.from_dict(session_dict)

    async def delete(self, session_id: str) -> None:
        """Delete session from Redis."""
        redis = await self._get_redis()
        if redis is None:
            return

        # Load session to get organization ID
        session = await self.load(session_id)
        if session:
            # Remove from organization index
            org_key = f"{self.key_prefix}org:{session.organization_id}"
            await redis.srem(org_key, session_id)

        key = f"{self.key_prefix}{session_id}"
        await redis.delete(key)

    async def list_active(
        self,
        organization_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[CallSession]:
        """List active sessions from Redis."""
        redis = await self._get_redis()
        if redis is None:
            return []

        sessions = []

        if organization_id:
            # Get sessions for specific organization
            org_key = f"{self.key_prefix}org:{organization_id}"
            session_ids = await redis.smembers(org_key)

            for session_id in session_ids:
                if len(sessions) >= limit:
                    break

                session = await self.load(session_id.decode())
                if session and session.state not in (
                    CallState.COMPLETED,
                    CallState.FAILED,
                    CallState.CANCELED,
                ):
                    sessions.append(session)
        else:
            # Scan all sessions (less efficient)
            cursor = 0
            while True:
                cursor, keys = await redis.scan(
                    cursor,
                    match=f"{self.key_prefix}*",
                    count=100,
                )

                for key in keys:
                    if len(sessions) >= limit:
                        break

                    key_str = key.decode()
                    if ":org:" in key_str:
                        continue  # Skip index keys

                    session_id = key_str.replace(self.key_prefix, "")
                    session = await self.load(session_id)

                    if session and session.state not in (
                        CallState.COMPLETED,
                        CallState.FAILED,
                        CallState.CANCELED,
                    ):
                        sessions.append(session)

                if cursor == 0 or len(sessions) >= limit:
                    break

        return sessions

    async def cleanup_expired(self, max_age_seconds: int) -> int:
        """Clean up expired sessions (Redis handles TTL automatically)."""
        # Redis automatically expires keys, but we can clean up indexes
        return 0


@dataclass
class SessionPoolConfig:
    """Configuration for session pooling."""

    max_sessions_per_org: int = 100
    max_total_sessions: int = 10000
    session_timeout_seconds: int = 3600  # 1 hour
    cleanup_interval_seconds: int = 60
    enable_session_reuse: bool = False


class SessionPool:
    """
    Manages a pool of active call sessions.

    Provides:
    - Session lifecycle management
    - Concurrency limits per organization
    - Automatic cleanup of stale sessions
    - Session metrics and monitoring
    """

    def __init__(
        self,
        storage: SessionStorageBackend,
        config: Optional[SessionPoolConfig] = None,
    ):
        """
        Initialize session pool.

        Args:
            storage: Storage backend for sessions
            config: Pool configuration
        """
        self.storage = storage
        self.config = config or SessionPoolConfig()

        # In-memory tracking for fast access
        self._active_sessions: Dict[str, CallSession] = {}
        self._org_sessions: Dict[str, Set[str]] = {}
        self._lock = asyncio.Lock()

        # Metrics
        self._total_created = 0
        self._total_completed = 0
        self._total_failed = 0

        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the session pool."""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Session pool started")

    async def stop(self) -> None:
        """Stop the session pool."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Session pool stopped")

    async def _cleanup_loop(self) -> None:
        """Background task to clean up expired sessions."""
        while self._running:
            try:
                await asyncio.sleep(self.config.cleanup_interval_seconds)
                await self._cleanup_stale_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Session cleanup error: {e}")

    async def _cleanup_stale_sessions(self) -> None:
        """Remove stale sessions from the pool."""
        now = datetime.utcnow()
        stale_sessions = []

        async with self._lock:
            for session_id, session in self._active_sessions.items():
                age = (now - session.started_at).total_seconds()
                if age > self.config.session_timeout_seconds:
                    stale_sessions.append(session_id)

        for session_id in stale_sessions:
            logger.warning(f"Cleaning up stale session: {session_id}")
            await self.release(session_id)

        # Also cleanup storage
        cleaned = await self.storage.cleanup_expired(
            self.config.session_timeout_seconds * 2
        )
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} expired sessions from storage")

    async def acquire(
        self,
        organization_id: str,
        agent_id: str,
        direction: CallDirection,
        from_number: str,
        to_number: str,
        config: Optional[CallConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CallSession:
        """
        Acquire a new session from the pool.

        Args:
            organization_id: Organization ID
            agent_id: Agent ID
            direction: Call direction
            from_number: From phone number
            to_number: To phone number
            config: Call configuration
            metadata: Additional metadata

        Returns:
            New call session

        Raises:
            RuntimeError: If pool limits exceeded
        """
        async with self._lock:
            # Check organization limit
            org_sessions = self._org_sessions.get(organization_id, set())
            if len(org_sessions) >= self.config.max_sessions_per_org:
                raise RuntimeError(
                    f"Organization {organization_id} has reached maximum "
                    f"concurrent sessions ({self.config.max_sessions_per_org})"
                )

            # Check total limit
            if len(self._active_sessions) >= self.config.max_total_sessions:
                raise RuntimeError(
                    f"Pool has reached maximum total sessions "
                    f"({self.config.max_total_sessions})"
                )

            # Create new session
            session = CallSession(
                id=f"call_{uuid.uuid4().hex}",
                organization_id=organization_id,
                agent_id=agent_id,
                direction=direction,
                from_number=from_number,
                to_number=to_number,
                config=config or CallConfig(),
                metadata=metadata or {},
            )

            # Track session
            self._active_sessions[session.id] = session
            if organization_id not in self._org_sessions:
                self._org_sessions[organization_id] = set()
            self._org_sessions[organization_id].add(session.id)

            self._total_created += 1

        # Persist to storage
        await self.storage.save(session)

        logger.info(
            f"Acquired session {session.id} for org {organization_id} "
            f"(active: {len(self._active_sessions)})"
        )

        return session

    async def release(self, session_id: str) -> None:
        """
        Release a session back to the pool.

        Args:
            session_id: Session ID to release
        """
        async with self._lock:
            session = self._active_sessions.pop(session_id, None)
            if session:
                # Remove from organization tracking
                org_sessions = self._org_sessions.get(session.organization_id)
                if org_sessions:
                    org_sessions.discard(session_id)

                # Update metrics
                if session.state == CallState.COMPLETED:
                    self._total_completed += 1
                elif session.state in (CallState.FAILED, CallState.CANCELED):
                    self._total_failed += 1

        # Remove from storage
        await self.storage.delete(session_id)

        logger.info(
            f"Released session {session_id} "
            f"(active: {len(self._active_sessions)})"
        )

    async def get(self, session_id: str) -> Optional[CallSession]:
        """
        Get a session by ID.

        Args:
            session_id: Session ID

        Returns:
            Call session or None if not found
        """
        # Check in-memory first
        session = self._active_sessions.get(session_id)
        if session:
            return session

        # Fall back to storage
        return await self.storage.load(session_id)

    async def update(self, session: CallSession) -> None:
        """
        Update a session.

        Args:
            session: Updated session
        """
        async with self._lock:
            if session.id in self._active_sessions:
                self._active_sessions[session.id] = session

        await self.storage.save(session)

    async def list_active(
        self,
        organization_id: Optional[str] = None,
    ) -> List[CallSession]:
        """
        List active sessions.

        Args:
            organization_id: Filter by organization

        Returns:
            List of active sessions
        """
        if organization_id:
            session_ids = self._org_sessions.get(organization_id, set())
            return [
                self._active_sessions[sid]
                for sid in session_ids
                if sid in self._active_sessions
            ]
        else:
            return list(self._active_sessions.values())

    def get_metrics(self) -> Dict[str, Any]:
        """Get pool metrics."""
        return {
            "active_sessions": len(self._active_sessions),
            "total_created": self._total_created,
            "total_completed": self._total_completed,
            "total_failed": self._total_failed,
            "organizations": len(self._org_sessions),
            "sessions_per_org": {
                org_id: len(sessions)
                for org_id, sessions in self._org_sessions.items()
            },
        }


class SessionLifecycleManager:
    """
    Manages the complete lifecycle of call sessions.

    Handles:
    - Session state transitions
    - Event emission
    - Metric tracking
    - Error handling
    """

    # Valid state transitions
    VALID_TRANSITIONS: Dict[CallState, Set[CallState]] = {
        CallState.INITIALIZING: {
            CallState.QUEUED,
            CallState.RINGING,
            CallState.FAILED,
            CallState.CANCELED,
        },
        CallState.QUEUED: {
            CallState.RINGING,
            CallState.FAILED,
            CallState.CANCELED,
        },
        CallState.RINGING: {
            CallState.CONNECTED,
            CallState.NO_ANSWER,
            CallState.BUSY,
            CallState.VOICEMAIL,
            CallState.FAILED,
            CallState.CANCELED,
        },
        CallState.CONNECTED: {
            CallState.AGENT_SPEAKING,
            CallState.CUSTOMER_SPEAKING,
            CallState.PROCESSING,
            CallState.SILENCE,
            CallState.TRANSFERRING,
            CallState.ON_HOLD,
            CallState.COMPLETED,
            CallState.FAILED,
        },
        CallState.AGENT_SPEAKING: {
            CallState.CUSTOMER_SPEAKING,
            CallState.PROCESSING,
            CallState.SILENCE,
            CallState.TRANSFERRING,
            CallState.ON_HOLD,
            CallState.COMPLETED,
            CallState.FAILED,
        },
        CallState.CUSTOMER_SPEAKING: {
            CallState.AGENT_SPEAKING,
            CallState.PROCESSING,
            CallState.SILENCE,
            CallState.TRANSFERRING,
            CallState.ON_HOLD,
            CallState.COMPLETED,
            CallState.FAILED,
        },
        CallState.PROCESSING: {
            CallState.AGENT_SPEAKING,
            CallState.CUSTOMER_SPEAKING,
            CallState.SILENCE,
            CallState.TRANSFERRING,
            CallState.ON_HOLD,
            CallState.COMPLETED,
            CallState.FAILED,
        },
        CallState.SILENCE: {
            CallState.AGENT_SPEAKING,
            CallState.CUSTOMER_SPEAKING,
            CallState.PROCESSING,
            CallState.TRANSFERRING,
            CallState.ON_HOLD,
            CallState.COMPLETED,
            CallState.FAILED,
        },
        CallState.TRANSFERRING: {
            CallState.CONNECTED,
            CallState.COMPLETED,
            CallState.FAILED,
        },
        CallState.ON_HOLD: {
            CallState.CONNECTED,
            CallState.AGENT_SPEAKING,
            CallState.CUSTOMER_SPEAKING,
            CallState.COMPLETED,
            CallState.FAILED,
        },
        CallState.VOICEMAIL: {
            CallState.AGENT_SPEAKING,
            CallState.COMPLETED,
            CallState.FAILED,
        },
        # Terminal states have no valid transitions
        CallState.COMPLETED: set(),
        CallState.FAILED: set(),
        CallState.NO_ANSWER: set(),
        CallState.BUSY: set(),
        CallState.CANCELED: set(),
    }

    def __init__(
        self,
        session_pool: SessionPool,
        event_callback: Optional[Callable[[OrchestratorEvent], Coroutine[Any, Any, None]]] = None,
    ):
        """
        Initialize lifecycle manager.

        Args:
            session_pool: Session pool
            event_callback: Callback for session events
        """
        self.session_pool = session_pool
        self.event_callback = event_callback

    async def create_session(
        self,
        organization_id: str,
        agent_id: str,
        direction: CallDirection,
        from_number: str,
        to_number: str,
        config: Optional[CallConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CallSession:
        """
        Create and initialize a new session.

        Args:
            organization_id: Organization ID
            agent_id: Agent ID
            direction: Call direction
            from_number: From phone number
            to_number: To phone number
            config: Call configuration
            metadata: Additional metadata

        Returns:
            New call session
        """
        session = await self.session_pool.acquire(
            organization_id=organization_id,
            agent_id=agent_id,
            direction=direction,
            from_number=from_number,
            to_number=to_number,
            config=config,
            metadata=metadata,
        )

        # Emit session created event
        await self._emit_event(session, EventType.CALL_STARTED, {
            "direction": direction.value,
            "from": from_number,
            "to": to_number,
        })

        return session

    async def transition_state(
        self,
        session: CallSession,
        new_state: CallState,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Transition session to a new state.

        Args:
            session: Call session
            new_state: Target state
            reason: Reason for transition
            metadata: Additional metadata

        Returns:
            True if transition was successful
        """
        current_state = session.state
        valid_transitions = self.VALID_TRANSITIONS.get(current_state, set())

        if new_state not in valid_transitions:
            logger.warning(
                f"Invalid state transition for session {session.id}: "
                f"{current_state.value} -> {new_state.value}"
            )
            return False

        # Update session state
        session.transition_state(new_state, reason)

        # Update metrics based on state
        self._update_metrics_for_state(session, new_state)

        # Persist update
        await self.session_pool.update(session)

        # Emit state change event
        await self._emit_event(session, EventType.STATE_CHANGED, {
            "previous_state": current_state.value,
            "new_state": new_state.value,
            "reason": reason,
            **(metadata or {}),
        })

        logger.debug(
            f"Session {session.id} transitioned: "
            f"{current_state.value} -> {new_state.value}"
        )

        return True

    def _update_metrics_for_state(
        self,
        session: CallSession,
        state: CallState,
    ) -> None:
        """Update session metrics based on state."""
        now = datetime.utcnow()

        if state == CallState.CONNECTED:
            session.connected_at = now
            session.metrics.time_to_connect_ms = (
                (now - session.started_at).total_seconds() * 1000
            )

        elif state == CallState.AGENT_SPEAKING:
            if session.metrics.time_to_first_response_ms == 0:
                session.metrics.time_to_first_response_ms = (
                    (now - session.connected_at).total_seconds() * 1000
                    if session.connected_at else 0
                )

        elif state in (
            CallState.COMPLETED,
            CallState.FAILED,
            CallState.NO_ANSWER,
            CallState.BUSY,
            CallState.CANCELED,
        ):
            session.ended_at = now
            if session.connected_at:
                duration = (now - session.connected_at).total_seconds()
                session.metrics.total_duration_seconds = duration

    async def end_session(
        self,
        session: CallSession,
        state: CallState = CallState.COMPLETED,
        reason: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        End a session.

        Args:
            session: Call session
            state: Terminal state
            reason: Reason for ending
            error: Error message if failed
        """
        # Transition to terminal state
        await self.transition_state(session, state, reason)

        # Emit appropriate event
        if state == CallState.COMPLETED:
            event_type = EventType.CALL_ENDED
        elif state == CallState.FAILED:
            event_type = EventType.CALL_FAILED
        else:
            event_type = EventType.CALL_ENDED

        await self._emit_event(session, event_type, {
            "reason": reason,
            "error": error,
            "duration_seconds": session.metrics.total_duration_seconds,
            "transcript_entries": len(session.transcript),
        })

        # Release session
        await self.session_pool.release(session.id)

    async def add_transcript_entry(
        self,
        session: CallSession,
        role: ParticipantRole,
        content: str,
        confidence: float = 1.0,
    ) -> None:
        """
        Add a transcript entry to the session.

        Args:
            session: Call session
            role: Speaker role
            content: Text content
            confidence: Transcription confidence
        """
        session.add_transcript_entry(role, content, confidence)
        await self.session_pool.update(session)

        await self._emit_event(session, EventType.TRANSCRIPT_UPDATED, {
            "role": role.value,
            "content": content,
            "entry_count": len(session.transcript),
        })

    async def _emit_event(
        self,
        session: CallSession,
        event_type: EventType,
        data: Dict[str, Any],
    ) -> None:
        """Emit a session event."""
        session.add_event(event_type, data)

        if self.event_callback:
            event = OrchestratorEvent(
                id=f"evt_{uuid.uuid4().hex[:12]}",
                session_id=session.id,
                event_type=event_type,
                data=data,
            )
            try:
                await self.event_callback(event)
            except Exception as e:
                logger.exception(f"Event callback failed: {e}")


@asynccontextmanager
async def session_context(
    lifecycle: SessionLifecycleManager,
    organization_id: str,
    agent_id: str,
    direction: CallDirection,
    from_number: str,
    to_number: str,
    config: Optional[CallConfig] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[CallSession]:
    """
    Context manager for call sessions.

    Automatically handles session creation and cleanup.

    Args:
        lifecycle: Lifecycle manager
        organization_id: Organization ID
        agent_id: Agent ID
        direction: Call direction
        from_number: From phone number
        to_number: To phone number
        config: Call configuration
        metadata: Additional metadata

    Yields:
        Call session

    Example:
        async with session_context(lifecycle, org, agent, ...) as session:
            # Process call
            await handle_call(session)
        # Session automatically released
    """
    session = await lifecycle.create_session(
        organization_id=organization_id,
        agent_id=agent_id,
        direction=direction,
        from_number=from_number,
        to_number=to_number,
        config=config,
        metadata=metadata,
    )

    try:
        yield session
    except Exception as e:
        # End session with failure state
        await lifecycle.end_session(
            session,
            state=CallState.FAILED,
            reason="Exception during call processing",
            error=str(e),
        )
        raise
    else:
        # End session normally if still active
        if session.state not in (
            CallState.COMPLETED,
            CallState.FAILED,
            CallState.NO_ANSWER,
            CallState.BUSY,
            CallState.CANCELED,
        ):
            await lifecycle.end_session(session, state=CallState.COMPLETED)


class SessionRecoveryManager:
    """
    Manages session recovery for fault tolerance.

    Handles:
    - Session checkpoint creation
    - Recovery from checkpoints
    - State reconstruction
    """

    def __init__(
        self,
        storage: SessionStorageBackend,
        checkpoint_interval_seconds: int = 5,
    ):
        """
        Initialize recovery manager.

        Args:
            storage: Storage backend
            checkpoint_interval_seconds: Checkpoint interval
        """
        self.storage = storage
        self.checkpoint_interval = checkpoint_interval_seconds
        self._checkpoint_tasks: Dict[str, asyncio.Task] = {}

    async def start_checkpointing(self, session: CallSession) -> None:
        """
        Start periodic checkpointing for a session.

        Args:
            session: Call session
        """
        if session.id in self._checkpoint_tasks:
            return

        task = asyncio.create_task(self._checkpoint_loop(session))
        self._checkpoint_tasks[session.id] = task
        logger.debug(f"Started checkpointing for session {session.id}")

    async def stop_checkpointing(self, session_id: str) -> None:
        """
        Stop checkpointing for a session.

        Args:
            session_id: Session ID
        """
        task = self._checkpoint_tasks.pop(session_id, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.debug(f"Stopped checkpointing for session {session_id}")

    async def _checkpoint_loop(self, session: CallSession) -> None:
        """Checkpoint loop for a session."""
        while True:
            try:
                await asyncio.sleep(self.checkpoint_interval)
                await self._create_checkpoint(session)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Checkpoint failed for session {session.id}: {e}")

    async def _create_checkpoint(self, session: CallSession) -> None:
        """Create a checkpoint for a session."""
        await self.storage.save(session)
        logger.debug(f"Checkpointed session {session.id}")

    async def recover_session(self, session_id: str) -> Optional[CallSession]:
        """
        Recover a session from storage.

        Args:
            session_id: Session ID

        Returns:
            Recovered session or None
        """
        session = await self.storage.load(session_id)
        if session:
            logger.info(f"Recovered session {session_id} in state {session.state}")
            # Start checkpointing for recovered session
            await self.start_checkpointing(session)
        return session

    async def list_recoverable_sessions(
        self,
        organization_id: Optional[str] = None,
    ) -> List[CallSession]:
        """
        List sessions that can be recovered.

        Args:
            organization_id: Filter by organization

        Returns:
            List of recoverable sessions
        """
        return await self.storage.list_active(organization_id)


__all__ = [
    "SessionStorageBackend",
    "InMemorySessionStorage",
    "RedisSessionStorage",
    "SessionPoolConfig",
    "SessionPool",
    "SessionLifecycleManager",
    "session_context",
    "SessionRecoveryManager",
]
