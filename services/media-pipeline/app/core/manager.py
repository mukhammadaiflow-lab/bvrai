"""Session manager - manages all active media sessions."""

import asyncio
import structlog
from typing import Optional, Dict, List, Any
from datetime import datetime
import time

from app.core.session import MediaSession, SessionConfig, SessionState
from app.config import settings


logger = structlog.get_logger()


class SessionManager:
    """
    Manages all active media sessions.

    Responsibilities:
    - Session lifecycle management
    - Resource allocation and limits
    - Session lookup and routing
    - Metrics aggregation
    - Cleanup of stale sessions
    """

    def __init__(self):
        self._sessions: Dict[str, MediaSession] = {}
        self._sessions_by_call: Dict[str, str] = {}  # call_id -> session_id
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._max_sessions = settings.max_concurrent_sessions

        # Metrics
        self._total_sessions_created = 0
        self._total_sessions_completed = 0
        self._peak_concurrent = 0

        logger.info(
            "session_manager_initialized",
            max_sessions=self._max_sessions,
        )

    async def start(self) -> None:
        """Start the session manager."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("session_manager_started")

    async def stop(self) -> None:
        """Stop the session manager and cleanup all sessions."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Disconnect all sessions
        async with self._lock:
            for session in list(self._sessions.values()):
                try:
                    await session.disconnect()
                except Exception as e:
                    logger.error(
                        "session_cleanup_error",
                        session_id=session.id,
                        error=str(e),
                    )

            self._sessions.clear()
            self._sessions_by_call.clear()

        logger.info("session_manager_stopped")

    async def create_session(self, config: SessionConfig) -> MediaSession:
        """
        Create a new media session.

        Args:
            config: Session configuration

        Returns:
            Created MediaSession

        Raises:
            RuntimeError: If max sessions reached
        """
        async with self._lock:
            # Check limits
            if len(self._sessions) >= self._max_sessions:
                logger.warning(
                    "max_sessions_reached",
                    current=len(self._sessions),
                    max=self._max_sessions,
                )
                raise RuntimeError("Maximum concurrent sessions reached")

            # Check for duplicate call
            if config.call_id in self._sessions_by_call:
                existing_session_id = self._sessions_by_call[config.call_id]
                logger.warning(
                    "duplicate_call_session",
                    call_id=config.call_id,
                    existing_session=existing_session_id,
                )
                # Return existing session
                return self._sessions[existing_session_id]

            # Create session
            session = MediaSession(config)
            self._sessions[session.id] = session
            self._sessions_by_call[config.call_id] = session.id

            # Update metrics
            self._total_sessions_created += 1
            self._peak_concurrent = max(self._peak_concurrent, len(self._sessions))

            logger.info(
                "session_created",
                session_id=session.id,
                call_id=config.call_id,
                total_active=len(self._sessions),
            )

            return session

    async def get_session(self, session_id: str) -> Optional[MediaSession]:
        """Get session by ID."""
        return self._sessions.get(session_id)

    async def get_session_by_call(self, call_id: str) -> Optional[MediaSession]:
        """Get session by call ID."""
        session_id = self._sessions_by_call.get(call_id)
        if session_id:
            return self._sessions.get(session_id)
        return None

    async def remove_session(self, session_id: str) -> bool:
        """
        Remove a session.

        Args:
            session_id: Session to remove

        Returns:
            True if session was removed
        """
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                self._sessions_by_call.pop(session.call_id, None)
                self._total_sessions_completed += 1

                logger.info(
                    "session_removed",
                    session_id=session_id,
                    total_active=len(self._sessions),
                )
                return True
            return False

    async def connect_session(self, session_id: str) -> bool:
        """Connect a session."""
        session = await self.get_session(session_id)
        if session:
            await session.connect()
            return True
        return False

    async def disconnect_session(self, session_id: str) -> bool:
        """Disconnect and remove a session."""
        session = await self.get_session(session_id)
        if session:
            await session.disconnect()
            await self.remove_session(session_id)
            return True
        return False

    async def list_sessions(
        self,
        state: Optional[SessionState] = None,
        agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all sessions with optional filtering.

        Args:
            state: Filter by state
            agent_id: Filter by agent

        Returns:
            List of session info dicts
        """
        sessions = []

        for session in self._sessions.values():
            if state and session.state != state:
                continue
            if agent_id and session.agent_id != agent_id:
                continue

            sessions.append({
                "session_id": session.id,
                "call_id": session.call_id,
                "agent_id": session.agent_id,
                "state": session.state.value,
                "source": session.config.source,
                "direction": session.config.direction,
                "duration_seconds": session.stats.duration_seconds,
            })

        return sessions

    def get_metrics(self) -> Dict[str, Any]:
        """Get manager metrics."""
        active_states = {}
        for session in self._sessions.values():
            state = session.state.value
            active_states[state] = active_states.get(state, 0) + 1

        return {
            "active_sessions": len(self._sessions),
            "peak_concurrent": self._peak_concurrent,
            "total_created": self._total_sessions_created,
            "total_completed": self._total_sessions_completed,
            "max_sessions": self._max_sessions,
            "utilization_percent": round(
                len(self._sessions) / self._max_sessions * 100, 2
            ),
            "sessions_by_state": active_states,
        }

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of stale sessions."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._cleanup_stale_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("cleanup_loop_error", error=str(e))

    async def _cleanup_stale_sessions(self) -> None:
        """Remove sessions that are stale or failed."""
        now = time.time()
        stale_timeout = 300  # 5 minutes

        stale_sessions = []

        async with self._lock:
            for session in self._sessions.values():
                # Check for failed sessions
                if session.state == SessionState.FAILED:
                    stale_sessions.append(session.id)
                    continue

                # Check for disconnected sessions not cleaned up
                if session.state == SessionState.DISCONNECTED:
                    stale_sessions.append(session.id)
                    continue

                # Check for very old pending sessions
                if session.state == SessionState.PENDING:
                    age = now - session.stats.created_at
                    if age > stale_timeout:
                        stale_sessions.append(session.id)

        # Remove stale sessions
        for session_id in stale_sessions:
            await self.disconnect_session(session_id)
            logger.info("stale_session_cleaned", session_id=session_id)
