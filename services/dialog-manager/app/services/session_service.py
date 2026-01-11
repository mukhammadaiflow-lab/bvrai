"""
Session Service - Manages conversation session state.

Handles:
- Session creation and retrieval
- Conversation history management
- Session cleanup/expiration
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import structlog

from app.config import get_settings

logger = structlog.get_logger()


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """Conversation session state."""

    session_id: str
    tenant_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    history: list[ConversationTurn] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    system_prompt: str = ""
    few_shots: list[dict[str, str]] = field(default_factory=list)


class SessionService:
    """
    Manages conversation sessions.

    Features:
    - In-memory session storage
    - History management with configurable limits
    - Session expiration
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._settings = get_settings()
        self._cleanup_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the session cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop the session cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_loop(self) -> None:
        """Periodically clean up expired sessions."""
        while True:
            await asyncio.sleep(60)  # Check every minute
            await self._cleanup_expired()

    async def _cleanup_expired(self) -> None:
        """Remove expired sessions."""
        now = datetime.utcnow()
        ttl = timedelta(seconds=self._settings.session_ttl_seconds)
        expired = []

        for session_id, session in self._sessions.items():
            if now - session.last_activity > ttl:
                expired.append(session_id)

        for session_id in expired:
            del self._sessions[session_id]
            logger.info("session_expired", session_id=session_id)

    def get_or_create(
        self,
        session_id: str,
        tenant_id: str,
        system_prompt: str = "",
        few_shots: list[dict[str, str]] | None = None,
    ) -> Session:
        """
        Get existing session or create new one.

        Args:
            session_id: Session identifier
            tenant_id: Tenant identifier
            system_prompt: System prompt for the session
            few_shots: Few-shot examples for the session

        Returns:
            Session instance
        """
        if session_id in self._sessions:
            session = self._sessions[session_id]
            session.last_activity = datetime.utcnow()
            return session

        session = Session(
            session_id=session_id,
            tenant_id=tenant_id,
            system_prompt=system_prompt,
            few_shots=few_shots or [],
        )
        self._sessions[session_id] = session

        logger.info(
            "session_created",
            session_id=session_id,
            tenant_id=tenant_id,
        )

        return session

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a conversation turn to the session.

        Args:
            session_id: Session identifier
            role: Role ("user" or "assistant")
            content: Turn content
            metadata: Optional metadata
        """
        session = self._sessions.get(session_id)
        if not session:
            logger.warning("session_not_found", session_id=session_id)
            return

        turn = ConversationTurn(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        session.history.append(turn)
        session.last_activity = datetime.utcnow()

        # Trim history if too long
        max_turns = self._settings.session_history_max_turns
        if len(session.history) > max_turns * 2:  # *2 for user+assistant pairs
            session.history = session.history[-(max_turns * 2) :]

    def get_history(self, session_id: str) -> list[ConversationTurn]:
        """Get conversation history for a session."""
        session = self._sessions.get(session_id)
        if not session:
            return []
        return session.history

    def get_session(self, session_id: str) -> Session | None:
        """Get session by ID."""
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("session_deleted", session_id=session_id)
            return True
        return False

    def update_context(self, session_id: str, context: dict[str, Any]) -> None:
        """Update session context."""
        session = self._sessions.get(session_id)
        if session:
            session.context.update(context)
            session.last_activity = datetime.utcnow()
