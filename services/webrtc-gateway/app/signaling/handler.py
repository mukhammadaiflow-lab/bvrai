"""WebSocket signaling handler."""

import asyncio
import json
from typing import Optional
from uuid import UUID

import structlog
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.config import get_settings
from app.signaling.models import (
    SignalingMessage,
    SignalingMessageType,
    ConnectPayload,
    ConnectedPayload,
    OfferPayload,
    AnswerPayload,
    IceCandidatePayload,
    ErrorPayload,
    TranscriptPayload,
)
from app.sessions.manager import SessionManager
from app.sessions.session import Session, SessionState

logger = structlog.get_logger()
settings = get_settings()


class SignalingHandler:
    """Handles WebSocket signaling for WebRTC connections."""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.logger = logger.bind(component="signaling")

    async def handle_connection(self, websocket: WebSocket) -> None:
        """Handle a new WebSocket connection."""
        await websocket.accept()

        session: Optional[Session] = None

        try:
            # Start heartbeat task
            heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(websocket)
            )

            while True:
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=settings.ws_heartbeat_interval * 2,
                    )

                    message = self._parse_message(data)
                    if not message:
                        continue

                    session = await self._handle_message(websocket, message, session)

                except asyncio.TimeoutError:
                    # Check if connection is still alive
                    try:
                        await self._send_message(
                            websocket,
                            SignalingMessage(type=SignalingMessageType.PING),
                        )
                    except Exception:
                        break

        except WebSocketDisconnect:
            self.logger.info("WebSocket disconnected")

        except Exception as e:
            self.logger.error("WebSocket error", error=str(e))

        finally:
            # Cleanup
            heartbeat_task.cancel()
            if session:
                await self.session_manager.end_session(session.session_id)

    async def _handle_message(
        self,
        websocket: WebSocket,
        message: SignalingMessage,
        session: Optional[Session],
    ) -> Optional[Session]:
        """Handle a signaling message."""
        self.logger.debug(
            "Received message",
            type=message.type.value,
            session_id=session.session_id if session else None,
        )

        handlers = {
            SignalingMessageType.CONNECT: self._handle_connect,
            SignalingMessageType.DISCONNECT: self._handle_disconnect,
            SignalingMessageType.OFFER: self._handle_offer,
            SignalingMessageType.ANSWER: self._handle_answer,
            SignalingMessageType.ICE_CANDIDATE: self._handle_ice_candidate,
            SignalingMessageType.SESSION_START: self._handle_session_start,
            SignalingMessageType.SESSION_END: self._handle_session_end,
            SignalingMessageType.PONG: self._handle_pong,
        }

        handler = handlers.get(message.type)
        if handler:
            return await handler(websocket, message, session)
        else:
            self.logger.warning("Unknown message type", type=message.type.value)
            return session

    async def _handle_connect(
        self,
        websocket: WebSocket,
        message: SignalingMessage,
        session: Optional[Session],
    ) -> Optional[Session]:
        """Handle connect message."""
        try:
            payload = ConnectPayload(**message.payload)
        except (TypeError, ValidationError) as e:
            await self._send_error(websocket, "INVALID_PAYLOAD", str(e))
            return session

        # Create session
        session = await self.session_manager.create_session(
            agent_id=payload.agent_id,
            websocket=websocket,
            api_key=payload.api_key,
            metadata=payload.metadata,
        )

        if not session:
            await self._send_error(
                websocket,
                "SESSION_CREATE_FAILED",
                "Failed to create session",
            )
            return None

        # Get ICE servers configuration
        ice_servers = self._get_ice_servers()

        # Send connected response
        await self._send_message(
            websocket,
            SignalingMessage(
                type=SignalingMessageType.CONNECTED,
                payload=ConnectedPayload(
                    session_id=session.session_id,
                    ice_servers=ice_servers,
                ).model_dump(),
            ),
        )

        self.logger.info(
            "Session connected",
            session_id=session.session_id,
            agent_id=str(payload.agent_id),
        )

        return session

    async def _handle_disconnect(
        self,
        websocket: WebSocket,
        message: SignalingMessage,
        session: Optional[Session],
    ) -> Optional[Session]:
        """Handle disconnect message."""
        if session:
            await self.session_manager.end_session(session.session_id)
            self.logger.info("Session disconnected", session_id=session.session_id)

        await websocket.close()
        return None

    async def _handle_offer(
        self,
        websocket: WebSocket,
        message: SignalingMessage,
        session: Optional[Session],
    ) -> Optional[Session]:
        """Handle SDP offer from client."""
        if not session:
            await self._send_error(websocket, "NO_SESSION", "No active session")
            return None

        try:
            payload = OfferPayload(**message.payload)
        except (TypeError, ValidationError) as e:
            await self._send_error(websocket, "INVALID_PAYLOAD", str(e))
            return session

        # Process offer and generate answer
        answer_sdp = await session.handle_offer(payload.sdp)

        if answer_sdp:
            await self._send_message(
                websocket,
                SignalingMessage(
                    type=SignalingMessageType.ANSWER,
                    payload=AnswerPayload(sdp=answer_sdp).model_dump(),
                ),
            )

        return session

    async def _handle_answer(
        self,
        websocket: WebSocket,
        message: SignalingMessage,
        session: Optional[Session],
    ) -> Optional[Session]:
        """Handle SDP answer from client."""
        if not session:
            await self._send_error(websocket, "NO_SESSION", "No active session")
            return None

        try:
            payload = AnswerPayload(**message.payload)
        except (TypeError, ValidationError) as e:
            await self._send_error(websocket, "INVALID_PAYLOAD", str(e))
            return session

        await session.handle_answer(payload.sdp)
        return session

    async def _handle_ice_candidate(
        self,
        websocket: WebSocket,
        message: SignalingMessage,
        session: Optional[Session],
    ) -> Optional[Session]:
        """Handle ICE candidate from client."""
        if not session:
            await self._send_error(websocket, "NO_SESSION", "No active session")
            return None

        try:
            payload = IceCandidatePayload(**message.payload)
        except (TypeError, ValidationError) as e:
            await self._send_error(websocket, "INVALID_PAYLOAD", str(e))
            return session

        await session.add_ice_candidate(
            payload.candidate,
            payload.sdp_mid,
            payload.sdp_m_line_index,
        )

        return session

    async def _handle_session_start(
        self,
        websocket: WebSocket,
        message: SignalingMessage,
        session: Optional[Session],
    ) -> Optional[Session]:
        """Handle session start (begin conversation)."""
        if not session:
            await self._send_error(websocket, "NO_SESSION", "No active session")
            return None

        if session.state != SessionState.CONNECTED:
            await self._send_error(
                websocket,
                "INVALID_STATE",
                f"Cannot start session in state {session.state.value}",
            )
            return session

        await session.start_conversation()

        await self._send_message(
            websocket,
            SignalingMessage(
                type=SignalingMessageType.SESSION_STARTED,
                payload={"session_id": session.session_id},
            ),
        )

        return session

    async def _handle_session_end(
        self,
        websocket: WebSocket,
        message: SignalingMessage,
        session: Optional[Session],
    ) -> Optional[Session]:
        """Handle session end."""
        if not session:
            return None

        await session.end_conversation()

        await self._send_message(
            websocket,
            SignalingMessage(
                type=SignalingMessageType.SESSION_ENDED,
                payload={"session_id": session.session_id},
            ),
        )

        return session

    async def _handle_pong(
        self,
        websocket: WebSocket,
        message: SignalingMessage,
        session: Optional[Session],
    ) -> Optional[Session]:
        """Handle pong response."""
        # Just update last activity timestamp
        if session:
            session.update_activity()
        return session

    def _parse_message(self, data: str) -> Optional[SignalingMessage]:
        """Parse incoming message."""
        try:
            parsed = json.loads(data)
            return SignalingMessage(**parsed)
        except (json.JSONDecodeError, ValidationError) as e:
            self.logger.warning("Failed to parse message", error=str(e))
            return None

    async def _send_message(
        self,
        websocket: WebSocket,
        message: SignalingMessage,
    ) -> None:
        """Send a message to the client."""
        try:
            await websocket.send_text(message.model_dump_json())
        except Exception as e:
            self.logger.error("Failed to send message", error=str(e))

    async def _send_error(
        self,
        websocket: WebSocket,
        code: str,
        message: str,
    ) -> None:
        """Send an error message."""
        await self._send_message(
            websocket,
            SignalingMessage(
                type=SignalingMessageType.ERROR,
                payload=ErrorPayload(code=code, message=message).model_dump(),
            ),
        )

    async def _heartbeat_loop(self, websocket: WebSocket) -> None:
        """Send periodic heartbeats."""
        while True:
            try:
                await asyncio.sleep(settings.ws_heartbeat_interval)
                await self._send_message(
                    websocket,
                    SignalingMessage(type=SignalingMessageType.PING),
                )
            except asyncio.CancelledError:
                break
            except Exception:
                break

    def _get_ice_servers(self) -> list[dict]:
        """Get ICE server configuration."""
        servers = []

        # Add STUN server
        if settings.stun_url:
            servers.append({"urls": settings.stun_url})

        # Add TURN server with credentials
        if settings.turn_enabled and settings.turn_url:
            import time
            import hmac
            import hashlib
            import base64

            # Generate time-limited credentials
            username = f"{int(time.time()) + 86400}:webrtc"
            credential = base64.b64encode(
                hmac.new(
                    settings.turn_secret.encode(),
                    username.encode(),
                    hashlib.sha1,
                ).digest()
            ).decode()

            servers.append({
                "urls": settings.turn_url,
                "username": username,
                "credential": credential,
            })

        return servers
