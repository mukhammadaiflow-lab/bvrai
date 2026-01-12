"""Session manager for WebRTC sessions."""

import asyncio
from typing import Optional
from uuid import UUID, uuid4

import httpx
import structlog
from fastapi import WebSocket

from app.config import get_settings
from app.sessions.session import Session, SessionState

logger = structlog.get_logger()
settings = get_settings()


class SessionManager:
    """Manages WebRTC sessions."""

    def __init__(self):
        self.sessions: dict[str, Session] = {}
        self.logger = logger.bind(component="session_manager")
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the session manager."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.logger.info("Session manager started")

    async def stop(self) -> None:
        """Stop the session manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # End all active sessions
        for session_id in list(self.sessions.keys()):
            await self.end_session(session_id)

        self.logger.info("Session manager stopped")

    async def create_session(
        self,
        agent_id: UUID,
        websocket: WebSocket,
        api_key: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[Session]:
        """Create a new session."""
        # Check capacity
        if len(self.sessions) >= settings.max_concurrent_sessions:
            self.logger.warning("Max sessions reached")
            return None

        # Validate agent exists
        agent_config = await self._get_agent_config(agent_id, api_key)
        if not agent_config:
            self.logger.warning("Agent not found", agent_id=str(agent_id))
            return None

        # Generate session ID
        session_id = str(uuid4())

        # Create session
        session = Session(
            session_id=session_id,
            agent_id=agent_id,
            websocket=websocket,
            api_key=api_key,
            metadata={
                **(metadata or {}),
                "agent_config": agent_config,
            },
        )

        self.sessions[session_id] = session

        self.logger.info(
            "Session created",
            session_id=session_id,
            agent_id=str(agent_id),
            total_sessions=len(self.sessions),
        )

        return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self.sessions.get(session_id)

    async def end_session(self, session_id: str) -> bool:
        """End a session."""
        session = self.sessions.pop(session_id, None)
        if not session:
            return False

        await session.end_conversation()

        # Notify platform API
        await self._notify_session_ended(session)

        self.logger.info(
            "Session ended",
            session_id=session_id,
            total_sessions=len(self.sessions),
        )

        return True

    def get_active_sessions(self) -> list[Session]:
        """Get all active sessions."""
        return [
            s for s in self.sessions.values()
            if s.state in [SessionState.CONNECTED, SessionState.ACTIVE]
        ]

    def get_session_count(self) -> int:
        """Get total session count."""
        return len(self.sessions)

    def get_active_session_count(self) -> int:
        """Get active session count."""
        return len(self.get_active_sessions())

    async def _get_agent_config(
        self,
        agent_id: UUID,
        api_key: Optional[str] = None,
    ) -> Optional[dict]:
        """Get agent configuration from Platform API."""
        try:
            async with httpx.AsyncClient() as client:
                headers = {}
                if api_key:
                    headers["X-API-Key"] = api_key

                response = await client.get(
                    f"{settings.platform_api_url}/api/v1/agents/{agent_id}",
                    headers=headers,
                    timeout=10.0,
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    self.logger.warning(
                        "Failed to get agent config",
                        agent_id=str(agent_id),
                        status=response.status_code,
                    )
                    return None

        except Exception as e:
            self.logger.error("Error getting agent config", error=str(e))
            # Return mock config for development
            return {
                "id": str(agent_id),
                "name": "Development Agent",
                "voice_id": "default",
                "language": "en-US",
                "greeting_message": "Hello! How can I help you today?",
                "system_prompt": "You are a helpful assistant.",
            }

    async def _notify_session_ended(self, session: Session) -> None:
        """Notify Platform API that session ended."""
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{settings.platform_api_url}/api/v1/webhooks/internal/call-event",
                    json={
                        "session_id": session.session_id,
                        "event_type": "call_ended",
                        "data": {
                            "duration_seconds": int(
                                (session.last_activity - session.created_at).total_seconds()
                            ),
                        },
                    },
                    headers={"X-Service-Key": "internal"},
                    timeout=5.0,
                )
        except Exception as e:
            self.logger.warning("Failed to notify session end", error=str(e))

    async def _cleanup_loop(self) -> None:
        """Periodically cleanup expired sessions."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Cleanup error", error=str(e))

    async def _cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions."""
        expired = [
            session_id
            for session_id, session in self.sessions.items()
            if session.is_expired()
        ]

        for session_id in expired:
            self.logger.info("Cleaning up expired session", session_id=session_id)
            await self.end_session(session_id)
