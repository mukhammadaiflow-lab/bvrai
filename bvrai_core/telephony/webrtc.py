"""
WebRTC Module

This module provides WebRTC support for browser-based voice calls
with ICE, SDP negotiation, and media handling.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

from .base import (
    Call,
    CallDirection,
    CallEvent,
    CallEventType,
    CallSession,
    CallState,
)


logger = logging.getLogger(__name__)


class WebRTCState(str, Enum):
    """WebRTC connection state."""
    NEW = "new"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    FAILED = "failed"
    CLOSED = "closed"


class ICEConnectionState(str, Enum):
    """ICE connection state."""
    NEW = "new"
    CHECKING = "checking"
    CONNECTED = "connected"
    COMPLETED = "completed"
    DISCONNECTED = "disconnected"
    FAILED = "failed"
    CLOSED = "closed"


class SignalingState(str, Enum):
    """Signaling state."""
    STABLE = "stable"
    HAVE_LOCAL_OFFER = "have-local-offer"
    HAVE_REMOTE_OFFER = "have-remote-offer"
    HAVE_LOCAL_PRANSWER = "have-local-pranswer"
    HAVE_REMOTE_PRANSWER = "have-remote-pranswer"
    CLOSED = "closed"


@dataclass
class ICECandidate:
    """ICE candidate."""

    candidate: str
    sdp_mid: Optional[str] = None
    sdp_mline_index: Optional[int] = None
    username_fragment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "candidate": self.candidate,
            "sdpMid": self.sdp_mid,
            "sdpMLineIndex": self.sdp_mline_index,
            "usernameFragment": self.username_fragment,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ICECandidate":
        """Create from dictionary."""
        return cls(
            candidate=data.get("candidate", ""),
            sdp_mid=data.get("sdpMid"),
            sdp_mline_index=data.get("sdpMLineIndex"),
            username_fragment=data.get("usernameFragment"),
        )


@dataclass
class SDPOffer:
    """SDP offer."""

    type: str = "offer"
    sdp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {"type": self.type, "sdp": self.sdp}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SDPOffer":
        """Create from dictionary."""
        return cls(type=data.get("type", "offer"), sdp=data.get("sdp", ""))


@dataclass
class SDPAnswer:
    """SDP answer."""

    type: str = "answer"
    sdp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {"type": self.type, "sdp": self.sdp}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SDPAnswer":
        """Create from dictionary."""
        return cls(type=data.get("type", "answer"), sdp=data.get("sdp", ""))


@dataclass
class WebRTCConfig:
    """WebRTC configuration."""

    # ICE servers
    ice_servers: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"urls": ["stun:stun.l.google.com:19302"]},
    ])

    # Audio settings
    audio_enabled: bool = True
    video_enabled: bool = False
    echo_cancellation: bool = True
    noise_suppression: bool = True
    auto_gain_control: bool = True

    # Codec preferences
    preferred_audio_codecs: List[str] = field(default_factory=lambda: ["opus", "pcmu", "pcma"])

    # ICE settings
    ice_transport_policy: str = "all"  # all, relay
    bundle_policy: str = "max-bundle"

    # Timeouts
    connection_timeout_seconds: int = 30
    ice_gathering_timeout_seconds: int = 10

    # TURN server (if needed)
    turn_server_url: Optional[str] = None
    turn_username: Optional[str] = None
    turn_credential: Optional[str] = None

    def get_ice_servers(self) -> List[Dict[str, Any]]:
        """Get complete ICE server configuration."""
        servers = list(self.ice_servers)

        if self.turn_server_url:
            servers.append({
                "urls": [self.turn_server_url],
                "username": self.turn_username,
                "credential": self.turn_credential,
            })

        return servers


@dataclass
class WebRTCConnection:
    """Represents a WebRTC connection."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # State
    state: WebRTCState = WebRTCState.NEW
    ice_state: ICEConnectionState = ICEConnectionState.NEW
    signaling_state: SignalingState = SignalingState.STABLE

    # SDP
    local_description: Optional[SDPOffer] = None
    remote_description: Optional[SDPAnswer] = None

    # ICE candidates
    local_candidates: List[ICECandidate] = field(default_factory=list)
    remote_candidates: List[ICECandidate] = field(default_factory=list)

    # Associated call
    call_id: Optional[str] = None
    session_id: Optional[str] = None

    # Media tracks
    local_audio_track: Optional[Any] = None
    remote_audio_track: Optional[Any] = None

    # Configuration
    config: WebRTCConfig = field(default_factory=WebRTCConfig)

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    connected_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None

    # Stats
    bytes_sent: int = 0
    bytes_received: int = 0
    packets_sent: int = 0
    packets_received: int = 0

    def is_connected(self) -> bool:
        """Check if connection is established."""
        return self.state == WebRTCState.CONNECTED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "state": self.state.value,
            "ice_state": self.ice_state.value,
            "signaling_state": self.signaling_state.value,
            "call_id": self.call_id,
            "created_at": self.created_at.isoformat(),
        }


class WebRTCSignaling:
    """
    WebRTC signaling handler.

    Handles SDP exchange and ICE candidate negotiation over WebSocket.
    """

    def __init__(self, config: Optional[WebRTCConfig] = None):
        """
        Initialize signaling handler.

        Args:
            config: WebRTC configuration
        """
        self.config = config or WebRTCConfig()
        self._connections: Dict[str, WebRTCConnection] = {}
        self._message_handlers: Dict[str, Callable] = {}

    async def handle_websocket(
        self,
        websocket: Any,
        connection_id: Optional[str] = None,
    ) -> WebRTCConnection:
        """
        Handle WebSocket connection for signaling.

        Args:
            websocket: WebSocket connection
            connection_id: Optional connection ID

        Returns:
            WebRTC connection object
        """
        # Create or get connection
        if connection_id and connection_id in self._connections:
            connection = self._connections[connection_id]
        else:
            connection = WebRTCConnection(config=self.config)
            self._connections[connection.id] = connection

        try:
            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "offer":
                    await self._handle_offer(connection, data, websocket)
                elif msg_type == "answer":
                    await self._handle_answer(connection, data, websocket)
                elif msg_type == "candidate":
                    await self._handle_candidate(connection, data, websocket)
                elif msg_type == "hangup":
                    await self._handle_hangup(connection, websocket)
                    break

        except Exception as e:
            logger.error(f"Signaling error: {e}")
            connection.state = WebRTCState.FAILED
        finally:
            if connection.state != WebRTCState.CONNECTED:
                connection.state = WebRTCState.CLOSED
                connection.closed_at = datetime.utcnow()

        return connection

    async def _handle_offer(
        self,
        connection: WebRTCConnection,
        data: Dict[str, Any],
        websocket: Any,
    ) -> None:
        """Handle incoming SDP offer."""
        offer = SDPOffer.from_dict(data)
        connection.remote_description = offer
        connection.signaling_state = SignalingState.HAVE_REMOTE_OFFER

        logger.info(f"Received SDP offer for connection {connection.id}")

        # Call handler if registered
        handler = self._message_handlers.get("offer")
        if handler:
            answer = await handler(connection, offer)
            if answer:
                connection.local_description = answer
                connection.signaling_state = SignalingState.STABLE

                await websocket.send(json.dumps({
                    "type": "answer",
                    "sdp": answer.sdp,
                }))

    async def _handle_answer(
        self,
        connection: WebRTCConnection,
        data: Dict[str, Any],
        websocket: Any,
    ) -> None:
        """Handle incoming SDP answer."""
        answer = SDPAnswer.from_dict(data)
        connection.remote_description = answer
        connection.signaling_state = SignalingState.STABLE

        logger.info(f"Received SDP answer for connection {connection.id}")

        handler = self._message_handlers.get("answer")
        if handler:
            await handler(connection, answer)

    async def _handle_candidate(
        self,
        connection: WebRTCConnection,
        data: Dict[str, Any],
        websocket: Any,
    ) -> None:
        """Handle incoming ICE candidate."""
        candidate_data = data.get("candidate", {})
        if isinstance(candidate_data, str):
            candidate = ICECandidate(candidate=candidate_data)
        else:
            candidate = ICECandidate.from_dict(candidate_data)

        connection.remote_candidates.append(candidate)

        logger.debug(f"Received ICE candidate for connection {connection.id}")

        handler = self._message_handlers.get("candidate")
        if handler:
            await handler(connection, candidate)

    async def _handle_hangup(
        self,
        connection: WebRTCConnection,
        websocket: Any,
    ) -> None:
        """Handle hangup signal."""
        connection.state = WebRTCState.CLOSED
        connection.closed_at = datetime.utcnow()

        logger.info(f"Connection {connection.id} hung up")

        handler = self._message_handlers.get("hangup")
        if handler:
            await handler(connection)

    async def send_offer(
        self,
        connection_id: str,
        offer: SDPOffer,
        websocket: Any,
    ) -> None:
        """Send SDP offer."""
        connection = self._connections.get(connection_id)
        if connection:
            connection.local_description = offer
            connection.signaling_state = SignalingState.HAVE_LOCAL_OFFER

        await websocket.send(json.dumps({
            "type": "offer",
            "sdp": offer.sdp,
        }))

    async def send_answer(
        self,
        connection_id: str,
        answer: SDPAnswer,
        websocket: Any,
    ) -> None:
        """Send SDP answer."""
        connection = self._connections.get(connection_id)
        if connection:
            connection.local_description = answer
            connection.signaling_state = SignalingState.STABLE

        await websocket.send(json.dumps({
            "type": "answer",
            "sdp": answer.sdp,
        }))

    async def send_candidate(
        self,
        connection_id: str,
        candidate: ICECandidate,
        websocket: Any,
    ) -> None:
        """Send ICE candidate."""
        connection = self._connections.get(connection_id)
        if connection:
            connection.local_candidates.append(candidate)

        await websocket.send(json.dumps({
            "type": "candidate",
            "candidate": candidate.to_dict(),
        }))

    def register_handler(self, event_type: str, handler: Callable) -> None:
        """Register a message handler."""
        self._message_handlers[event_type] = handler

    def get_connection(self, connection_id: str) -> Optional[WebRTCConnection]:
        """Get connection by ID."""
        return self._connections.get(connection_id)


class WebRTCManager:
    """
    Manages WebRTC connections and media.

    Coordinates signaling, ICE, and audio streaming.
    """

    def __init__(self, config: Optional[WebRTCConfig] = None):
        """
        Initialize WebRTC manager.

        Args:
            config: WebRTC configuration
        """
        self.config = config or WebRTCConfig()
        self._signaling = WebRTCSignaling(config)
        self._connections: Dict[str, WebRTCConnection] = {}
        self._audio_handlers: Dict[str, Callable[[bytes], None]] = {}

    async def create_connection(
        self,
        call_id: Optional[str] = None,
    ) -> WebRTCConnection:
        """
        Create a new WebRTC connection.

        Args:
            call_id: Associated call ID

        Returns:
            New connection object
        """
        connection = WebRTCConnection(
            config=self.config,
            call_id=call_id,
        )

        self._connections[connection.id] = connection

        logger.info(f"Created WebRTC connection: {connection.id}")

        return connection

    async def create_offer(
        self,
        connection_id: str,
    ) -> Optional[SDPOffer]:
        """
        Create an SDP offer for a connection.

        Args:
            connection_id: Connection ID

        Returns:
            SDP offer or None
        """
        connection = self._connections.get(connection_id)
        if not connection:
            return None

        # Build SDP offer
        sdp = self._build_offer_sdp(connection)

        offer = SDPOffer(sdp=sdp)
        connection.local_description = offer
        connection.signaling_state = SignalingState.HAVE_LOCAL_OFFER

        return offer

    async def handle_offer(
        self,
        connection_id: str,
        offer: SDPOffer,
    ) -> Optional[SDPAnswer]:
        """
        Handle an incoming SDP offer.

        Args:
            connection_id: Connection ID
            offer: SDP offer

        Returns:
            SDP answer or None
        """
        connection = self._connections.get(connection_id)
        if not connection:
            connection = await self.create_connection()

        connection.remote_description = offer
        connection.signaling_state = SignalingState.HAVE_REMOTE_OFFER

        # Build SDP answer
        sdp = self._build_answer_sdp(connection, offer)

        answer = SDPAnswer(sdp=sdp)
        connection.local_description = answer
        connection.signaling_state = SignalingState.STABLE

        return answer

    async def handle_answer(
        self,
        connection_id: str,
        answer: SDPAnswer,
    ) -> None:
        """
        Handle an incoming SDP answer.

        Args:
            connection_id: Connection ID
            answer: SDP answer
        """
        connection = self._connections.get(connection_id)
        if not connection:
            return

        connection.remote_description = answer
        connection.signaling_state = SignalingState.STABLE

    async def add_ice_candidate(
        self,
        connection_id: str,
        candidate: ICECandidate,
    ) -> None:
        """
        Add a remote ICE candidate.

        Args:
            connection_id: Connection ID
            candidate: ICE candidate
        """
        connection = self._connections.get(connection_id)
        if not connection:
            return

        connection.remote_candidates.append(candidate)

        # Check if ICE gathering is complete
        if candidate.candidate == "":
            connection.ice_state = ICEConnectionState.COMPLETED

    async def send_audio(
        self,
        connection_id: str,
        audio_data: bytes,
    ) -> None:
        """
        Send audio data to a connection.

        Args:
            connection_id: Connection ID
            audio_data: Audio bytes
        """
        connection = self._connections.get(connection_id)
        if not connection or not connection.is_connected():
            return

        # In a real implementation, this would send via RTP
        connection.bytes_sent += len(audio_data)
        connection.packets_sent += 1

    def register_audio_handler(
        self,
        connection_id: str,
        handler: Callable[[bytes], None],
    ) -> None:
        """
        Register handler for received audio.

        Args:
            connection_id: Connection ID
            handler: Audio callback
        """
        self._audio_handlers[connection_id] = handler

    async def close_connection(self, connection_id: str) -> None:
        """
        Close a WebRTC connection.

        Args:
            connection_id: Connection ID
        """
        connection = self._connections.get(connection_id)
        if not connection:
            return

        connection.state = WebRTCState.CLOSED
        connection.closed_at = datetime.utcnow()

        if connection_id in self._audio_handlers:
            del self._audio_handlers[connection_id]

        logger.info(f"Closed WebRTC connection: {connection_id}")

    def get_connection(self, connection_id: str) -> Optional[WebRTCConnection]:
        """Get connection by ID."""
        return self._connections.get(connection_id)

    def get_stats(self, connection_id: str) -> Dict[str, Any]:
        """Get connection statistics."""
        connection = self._connections.get(connection_id)
        if not connection:
            return {}

        return {
            "state": connection.state.value,
            "ice_state": connection.ice_state.value,
            "bytes_sent": connection.bytes_sent,
            "bytes_received": connection.bytes_received,
            "packets_sent": connection.packets_sent,
            "packets_received": connection.packets_received,
        }

    def _build_offer_sdp(self, connection: WebRTCConnection) -> str:
        """Build SDP offer string."""
        # Simplified SDP for audio
        sdp_lines = [
            "v=0",
            f"o=- {connection.id[:8]} 2 IN IP4 127.0.0.1",
            "s=-",
            "t=0 0",
            "a=group:BUNDLE 0",
            "a=msid-semantic: WMS",
            "m=audio 9 UDP/TLS/RTP/SAVPF 111 0 8",
            "c=IN IP4 0.0.0.0",
            "a=rtcp:9 IN IP4 0.0.0.0",
            "a=setup:actpass",
            "a=mid:0",
            "a=sendrecv",
            "a=rtcp-mux",
            "a=rtpmap:111 opus/48000/2",
            "a=rtpmap:0 PCMU/8000",
            "a=rtpmap:8 PCMA/8000",
            "a=fmtp:111 minptime=10;useinbandfec=1",
        ]

        return "\r\n".join(sdp_lines) + "\r\n"

    def _build_answer_sdp(self, connection: WebRTCConnection, offer: SDPOffer) -> str:
        """Build SDP answer string."""
        # Simplified SDP answer
        sdp_lines = [
            "v=0",
            f"o=- {connection.id[:8]} 2 IN IP4 127.0.0.1",
            "s=-",
            "t=0 0",
            "a=group:BUNDLE 0",
            "m=audio 9 UDP/TLS/RTP/SAVPF 111 0 8",
            "c=IN IP4 0.0.0.0",
            "a=rtcp:9 IN IP4 0.0.0.0",
            "a=setup:active",
            "a=mid:0",
            "a=recvonly",
            "a=rtcp-mux",
            "a=rtpmap:111 opus/48000/2",
            "a=rtpmap:0 PCMU/8000",
            "a=rtpmap:8 PCMA/8000",
        ]

        return "\r\n".join(sdp_lines) + "\r\n"


__all__ = [
    # Enums
    "WebRTCState",
    "ICEConnectionState",
    "SignalingState",
    # Data classes
    "ICECandidate",
    "SDPOffer",
    "SDPAnswer",
    "WebRTCConfig",
    "WebRTCConnection",
    # Classes
    "WebRTCSignaling",
    "WebRTCManager",
]
