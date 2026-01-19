"""
WebRTC Handler

WebRTC support for browser-based calls:
- SDP negotiation
- ICE candidate handling
- Peer connection management
- Media track handling
- Signaling server
"""

from typing import Optional, Dict, Any, List, Callable, Awaitable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging
import uuid
import json
import hashlib

logger = logging.getLogger(__name__)


class RTCConnectionState(str, Enum):
    """WebRTC connection states."""
    NEW = "new"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    FAILED = "failed"
    CLOSED = "closed"


class RTCSignalingState(str, Enum):
    """WebRTC signaling states."""
    STABLE = "stable"
    HAVE_LOCAL_OFFER = "have-local-offer"
    HAVE_REMOTE_OFFER = "have-remote-offer"
    HAVE_LOCAL_PRANSWER = "have-local-pranswer"
    HAVE_REMOTE_PRANSWER = "have-remote-pranswer"
    CLOSED = "closed"


class RTCIceConnectionState(str, Enum):
    """ICE connection states."""
    NEW = "new"
    CHECKING = "checking"
    CONNECTED = "connected"
    COMPLETED = "completed"
    FAILED = "failed"
    DISCONNECTED = "disconnected"
    CLOSED = "closed"


class MediaType(str, Enum):
    """Media types."""
    AUDIO = "audio"
    VIDEO = "video"
    DATA = "data"


@dataclass
class WebRTCConfig:
    """WebRTC configuration."""
    # ICE servers
    stun_servers: List[str] = field(default_factory=lambda: [
        "stun:stun.l.google.com:19302",
        "stun:stun1.l.google.com:19302",
    ])
    turn_servers: List[Dict[str, str]] = field(default_factory=list)

    # Timeouts
    ice_gathering_timeout_ms: int = 10000
    connection_timeout_ms: int = 30000
    negotiation_timeout_ms: int = 10000

    # Audio settings
    audio_codec: str = "opus"
    audio_sample_rate: int = 48000
    audio_channels: int = 1
    audio_bitrate: int = 32000

    # Features
    enable_dtls: bool = True
    enable_srtp: bool = True
    enable_rtcp_mux: bool = True
    enable_bundle: bool = True

    # Limits
    max_peers_per_session: int = 10
    max_concurrent_connections: int = 1000


@dataclass
class ICECandidate:
    """ICE candidate representation."""
    candidate: str
    sdp_mid: str
    sdp_m_line_index: int
    username_fragment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "candidate": self.candidate,
            "sdpMid": self.sdp_mid,
            "sdpMLineIndex": self.sdp_m_line_index,
            "usernameFragment": self.username_fragment,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ICECandidate":
        """Create from dictionary."""
        return cls(
            candidate=data.get("candidate", ""),
            sdp_mid=data.get("sdpMid", ""),
            sdp_m_line_index=data.get("sdpMLineIndex", 0),
            username_fragment=data.get("usernameFragment"),
        )


@dataclass
class SDPOffer:
    """SDP offer representation."""
    sdp: str
    type: str = "offer"
    session_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "sdp": self.sdp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SDPOffer":
        """Create from dictionary."""
        return cls(
            sdp=data.get("sdp", ""),
            type=data.get("type", "offer"),
        )


@dataclass
class SDPAnswer:
    """SDP answer representation."""
    sdp: str
    type: str = "answer"
    session_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "sdp": self.sdp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SDPAnswer":
        """Create from dictionary."""
        return cls(
            sdp=data.get("sdp", ""),
            type=data.get("type", "answer"),
        )


@dataclass
class MediaTrack:
    """Media track representation."""
    track_id: str
    kind: MediaType
    label: str = ""
    enabled: bool = True
    muted: bool = False

    # Track statistics
    bytes_sent: int = 0
    bytes_received: int = 0
    packets_sent: int = 0
    packets_received: int = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get track statistics."""
        return {
            "track_id": self.track_id,
            "kind": self.kind.value,
            "enabled": self.enabled,
            "muted": self.muted,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
        }


@dataclass
class PeerConnection:
    """WebRTC peer connection."""
    connection_id: str
    session_id: str
    peer_id: str

    # State
    connection_state: RTCConnectionState = RTCConnectionState.NEW
    signaling_state: RTCSignalingState = RTCSignalingState.STABLE
    ice_connection_state: RTCIceConnectionState = RTCIceConnectionState.NEW

    # SDP
    local_description: Optional[str] = None
    remote_description: Optional[str] = None

    # ICE candidates
    local_candidates: List[ICECandidate] = field(default_factory=list)
    remote_candidates: List[ICECandidate] = field(default_factory=list)
    ice_gathering_complete: bool = False

    # Media tracks
    local_tracks: Dict[str, MediaTrack] = field(default_factory=dict)
    remote_tracks: Dict[str, MediaTrack] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    connected_at: Optional[datetime] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_local_track(self, kind: MediaType, label: str = "") -> MediaTrack:
        """Add local media track."""
        track = MediaTrack(
            track_id=str(uuid.uuid4()),
            kind=kind,
            label=label,
        )
        self.local_tracks[track.track_id] = track
        return track

    def add_remote_track(self, kind: MediaType, label: str = "") -> MediaTrack:
        """Add remote media track."""
        track = MediaTrack(
            track_id=str(uuid.uuid4()),
            kind=kind,
            label=label,
        )
        self.remote_tracks[track.track_id] = track
        return track

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "connection_id": self.connection_id,
            "connection_state": self.connection_state.value,
            "ice_connection_state": self.ice_connection_state.value,
            "local_candidates": len(self.local_candidates),
            "remote_candidates": len(self.remote_candidates),
            "local_tracks": {
                tid: track.get_stats()
                for tid, track in self.local_tracks.items()
            },
            "remote_tracks": {
                tid: track.get_stats()
                for tid, track in self.remote_tracks.items()
            },
        }


class SDPParser:
    """SDP parser and modifier."""

    @staticmethod
    def parse(sdp: str) -> Dict[str, Any]:
        """Parse SDP into structured format."""
        result = {
            "version": 0,
            "origin": {},
            "session_name": "",
            "timing": {},
            "media": [],
            "attributes": [],
        }

        current_media = None

        for line in sdp.split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith("v="):
                result["version"] = int(line[2:])

            elif line.startswith("o="):
                parts = line[2:].split()
                if len(parts) >= 6:
                    result["origin"] = {
                        "username": parts[0],
                        "session_id": parts[1],
                        "session_version": parts[2],
                        "net_type": parts[3],
                        "address_type": parts[4],
                        "address": parts[5],
                    }

            elif line.startswith("s="):
                result["session_name"] = line[2:]

            elif line.startswith("t="):
                parts = line[2:].split()
                result["timing"] = {
                    "start": int(parts[0]) if parts else 0,
                    "stop": int(parts[1]) if len(parts) > 1 else 0,
                }

            elif line.startswith("m="):
                parts = line[2:].split()
                current_media = {
                    "type": parts[0] if parts else "",
                    "port": int(parts[1]) if len(parts) > 1 else 0,
                    "protocol": parts[2] if len(parts) > 2 else "",
                    "formats": parts[3:] if len(parts) > 3 else [],
                    "attributes": [],
                }
                result["media"].append(current_media)

            elif line.startswith("a="):
                attr = line[2:]
                if current_media:
                    current_media["attributes"].append(attr)
                else:
                    result["attributes"].append(attr)

        return result

    @staticmethod
    def modify_codec(sdp: str, codec: str, bitrate: int) -> str:
        """Modify SDP to prefer specific codec."""
        lines = sdp.split("\n")
        modified = []

        for line in lines:
            if line.startswith("a=rtpmap:") and codec.lower() in line.lower():
                # Found codec line
                modified.append(line)
                # Add bitrate constraint
                payload_type = line.split(":")[1].split()[0]
                modified.append(f"a=fmtp:{payload_type} maxaveragebitrate={bitrate}")
            else:
                modified.append(line)

        return "\n".join(modified)

    @staticmethod
    def extract_ice_credentials(sdp: str) -> Dict[str, str]:
        """Extract ICE credentials from SDP."""
        credentials = {}

        for line in sdp.split("\n"):
            if line.startswith("a=ice-ufrag:"):
                credentials["ice_ufrag"] = line.split(":")[1].strip()
            elif line.startswith("a=ice-pwd:"):
                credentials["ice_pwd"] = line.split(":")[1].strip()

        return credentials


class WebRTCHandler:
    """
    WebRTC connection handler.

    Manages WebRTC peer connections:
    - SDP offer/answer exchange
    - ICE candidate handling
    - Media track management
    """

    def __init__(self, config: Optional[WebRTCConfig] = None):
        self.config = config or WebRTCConfig()

        # Connections by session
        self._connections: Dict[str, Dict[str, PeerConnection]] = {}
        self._lock = asyncio.Lock()

        # Event callbacks
        self._on_ice_candidate: List[Callable[[str, ICECandidate], Awaitable[None]]] = []
        self._on_track: List[Callable[[str, MediaTrack], Awaitable[None]]] = []
        self._on_state_change: List[Callable[[str, RTCConnectionState], Awaitable[None]]] = []

    async def create_connection(
        self,
        session_id: str,
        peer_id: str,
    ) -> PeerConnection:
        """Create new peer connection."""
        async with self._lock:
            if session_id not in self._connections:
                self._connections[session_id] = {}

            connection = PeerConnection(
                connection_id=str(uuid.uuid4()),
                session_id=session_id,
                peer_id=peer_id,
            )

            # Add default audio track
            connection.add_local_track(MediaType.AUDIO, "voice")

            self._connections[session_id][peer_id] = connection
            return connection

    async def get_connection(
        self,
        session_id: str,
        peer_id: str,
    ) -> Optional[PeerConnection]:
        """Get peer connection."""
        session_connections = self._connections.get(session_id, {})
        return session_connections.get(peer_id)

    async def create_offer(
        self,
        session_id: str,
        peer_id: str,
    ) -> Optional[SDPOffer]:
        """Create SDP offer."""
        connection = await self.get_connection(session_id, peer_id)
        if not connection:
            return None

        # Generate SDP offer
        sdp = self._generate_offer_sdp(connection)
        offer = SDPOffer(sdp=sdp, session_id=session_id)

        connection.local_description = sdp
        connection.signaling_state = RTCSignalingState.HAVE_LOCAL_OFFER

        return offer

    async def create_answer(
        self,
        session_id: str,
        peer_id: str,
        offer: SDPOffer,
    ) -> Optional[SDPAnswer]:
        """Create SDP answer for offer."""
        connection = await self.get_connection(session_id, peer_id)
        if not connection:
            return None

        # Store remote description
        connection.remote_description = offer.sdp
        connection.signaling_state = RTCSignalingState.HAVE_REMOTE_OFFER

        # Generate answer
        sdp = self._generate_answer_sdp(connection, offer.sdp)
        answer = SDPAnswer(sdp=sdp, session_id=session_id)

        connection.local_description = sdp
        connection.signaling_state = RTCSignalingState.STABLE

        return answer

    async def set_remote_description(
        self,
        session_id: str,
        peer_id: str,
        sdp: str,
        sdp_type: str,
    ) -> bool:
        """Set remote description."""
        connection = await self.get_connection(session_id, peer_id)
        if not connection:
            return False

        connection.remote_description = sdp

        if sdp_type == "answer":
            connection.signaling_state = RTCSignalingState.STABLE

        return True

    async def add_ice_candidate(
        self,
        session_id: str,
        peer_id: str,
        candidate: ICECandidate,
    ) -> bool:
        """Add remote ICE candidate."""
        connection = await self.get_connection(session_id, peer_id)
        if not connection:
            return False

        connection.remote_candidates.append(candidate)

        # Process candidate
        await self._process_ice_candidate(connection, candidate)

        return True

    async def _process_ice_candidate(
        self,
        connection: PeerConnection,
        candidate: ICECandidate,
    ) -> None:
        """Process ICE candidate."""
        # In production: use aiortc or similar library
        # This is a framework implementation

        # Check if we can establish connection
        if (
            connection.ice_connection_state == RTCIceConnectionState.NEW
            and len(connection.remote_candidates) > 0
        ):
            connection.ice_connection_state = RTCIceConnectionState.CHECKING

        # Simulate connection establishment
        if len(connection.remote_candidates) >= 2:
            connection.ice_connection_state = RTCIceConnectionState.CONNECTED
            connection.connection_state = RTCConnectionState.CONNECTED
            connection.connected_at = datetime.utcnow()

            # Trigger state change callbacks
            for callback in self._on_state_change:
                try:
                    await callback(connection.connection_id, connection.connection_state)
                except Exception as e:
                    logger.error(f"State change callback error: {e}")

    async def close_connection(
        self,
        session_id: str,
        peer_id: str,
    ) -> bool:
        """Close peer connection."""
        async with self._lock:
            session_connections = self._connections.get(session_id, {})
            connection = session_connections.pop(peer_id, None)

            if connection:
                connection.connection_state = RTCConnectionState.CLOSED
                connection.ice_connection_state = RTCIceConnectionState.CLOSED
                connection.signaling_state = RTCSignalingState.CLOSED
                return True

            return False

    async def close_session(self, session_id: str) -> int:
        """Close all connections for session."""
        async with self._lock:
            session_connections = self._connections.pop(session_id, {})

            for connection in session_connections.values():
                connection.connection_state = RTCConnectionState.CLOSED
                connection.ice_connection_state = RTCIceConnectionState.CLOSED

            return len(session_connections)

    def on_ice_candidate(
        self,
        callback: Callable[[str, ICECandidate], Awaitable[None]],
    ) -> None:
        """Register ICE candidate callback."""
        self._on_ice_candidate.append(callback)

    def on_track(
        self,
        callback: Callable[[str, MediaTrack], Awaitable[None]],
    ) -> None:
        """Register track callback."""
        self._on_track.append(callback)

    def on_state_change(
        self,
        callback: Callable[[str, RTCConnectionState], Awaitable[None]],
    ) -> None:
        """Register state change callback."""
        self._on_state_change.append(callback)

    def _generate_offer_sdp(self, connection: PeerConnection) -> str:
        """Generate SDP offer."""
        session_id = str(int(datetime.utcnow().timestamp()))
        session_version = "1"

        sdp_lines = [
            "v=0",
            f"o=- {session_id} {session_version} IN IP4 0.0.0.0",
            "s=Voice Session",
            "t=0 0",
            "a=group:BUNDLE 0",
            "a=msid-semantic: WMS",
        ]

        # Add audio media section
        sdp_lines.extend([
            "m=audio 9 UDP/TLS/RTP/SAVPF 111 0 8",
            "c=IN IP4 0.0.0.0",
            "a=rtcp:9 IN IP4 0.0.0.0",
            f"a=ice-ufrag:{self._generate_ice_ufrag()}",
            f"a=ice-pwd:{self._generate_ice_pwd()}",
            "a=ice-options:trickle",
            "a=fingerprint:sha-256 " + self._generate_fingerprint(),
            "a=setup:actpass",
            "a=mid:0",
            "a=sendrecv",
            "a=rtcp-mux",
            "a=rtpmap:111 opus/48000/2",
            "a=fmtp:111 minptime=10;useinbandfec=1",
            "a=rtpmap:0 PCMU/8000",
            "a=rtpmap:8 PCMA/8000",
        ])

        return "\r\n".join(sdp_lines) + "\r\n"

    def _generate_answer_sdp(self, connection: PeerConnection, offer_sdp: str) -> str:
        """Generate SDP answer."""
        # Parse offer
        parsed = SDPParser.parse(offer_sdp)

        session_id = str(int(datetime.utcnow().timestamp()))
        session_version = "1"

        sdp_lines = [
            "v=0",
            f"o=- {session_id} {session_version} IN IP4 0.0.0.0",
            "s=Voice Session",
            "t=0 0",
            "a=group:BUNDLE 0",
        ]

        # Add audio media section matching offer
        sdp_lines.extend([
            "m=audio 9 UDP/TLS/RTP/SAVPF 111 0 8",
            "c=IN IP4 0.0.0.0",
            f"a=ice-ufrag:{self._generate_ice_ufrag()}",
            f"a=ice-pwd:{self._generate_ice_pwd()}",
            "a=ice-options:trickle",
            "a=fingerprint:sha-256 " + self._generate_fingerprint(),
            "a=setup:active",
            "a=mid:0",
            "a=sendrecv",
            "a=rtcp-mux",
            "a=rtpmap:111 opus/48000/2",
            "a=fmtp:111 minptime=10;useinbandfec=1",
        ])

        return "\r\n".join(sdp_lines) + "\r\n"

    def _generate_ice_ufrag(self) -> str:
        """Generate ICE username fragment."""
        return uuid.uuid4().hex[:8]

    def _generate_ice_pwd(self) -> str:
        """Generate ICE password."""
        return uuid.uuid4().hex[:24]

    def _generate_fingerprint(self) -> str:
        """Generate DTLS fingerprint."""
        # In production: use actual certificate fingerprint
        data = uuid.uuid4().bytes
        digest = hashlib.sha256(data).hexdigest().upper()
        return ":".join(digest[i:i+2] for i in range(0, 64, 2))

    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics."""
        total_connections = sum(
            len(conns) for conns in self._connections.values()
        )

        return {
            "total_sessions": len(self._connections),
            "total_connections": total_connections,
        }


class SignalingMessage:
    """Signaling message types."""
    OFFER = "offer"
    ANSWER = "answer"
    CANDIDATE = "candidate"
    JOIN = "join"
    LEAVE = "leave"
    ERROR = "error"


@dataclass
class SignalingEvent:
    """Signaling event."""
    type: str
    session_id: str
    peer_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "sessionId": self.session_id,
            "peerId": self.peer_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


class SignalingServer:
    """
    WebRTC signaling server.

    Handles:
    - Session management
    - Message routing
    - Peer discovery
    """

    def __init__(self):
        # Sessions and peers
        self._sessions: Dict[str, Set[str]] = {}
        self._peer_queues: Dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()

        # Event handlers
        self._handlers: Dict[str, List[Callable[[SignalingEvent], Awaitable[None]]]] = {}

    async def join(self, session_id: str, peer_id: str) -> asyncio.Queue:
        """Join signaling session."""
        async with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = set()

            self._sessions[session_id].add(peer_id)

            # Create message queue for peer
            queue: asyncio.Queue = asyncio.Queue()
            self._peer_queues[peer_id] = queue

            # Notify other peers
            event = SignalingEvent(
                type=SignalingMessage.JOIN,
                session_id=session_id,
                peer_id=peer_id,
            )
            await self._broadcast(session_id, peer_id, event)

            return queue

    async def leave(self, session_id: str, peer_id: str) -> None:
        """Leave signaling session."""
        async with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].discard(peer_id)

                if not self._sessions[session_id]:
                    del self._sessions[session_id]

            self._peer_queues.pop(peer_id, None)

        # Notify other peers
        event = SignalingEvent(
            type=SignalingMessage.LEAVE,
            session_id=session_id,
            peer_id=peer_id,
        )
        await self._broadcast(session_id, peer_id, event)

    async def send(
        self,
        session_id: str,
        from_peer: str,
        to_peer: str,
        message_type: str,
        data: Dict[str, Any],
    ) -> bool:
        """Send message to specific peer."""
        queue = self._peer_queues.get(to_peer)
        if not queue:
            return False

        event = SignalingEvent(
            type=message_type,
            session_id=session_id,
            peer_id=from_peer,
            data=data,
        )

        await queue.put(event)
        return True

    async def broadcast(
        self,
        session_id: str,
        from_peer: str,
        message_type: str,
        data: Dict[str, Any],
    ) -> int:
        """Broadcast message to all peers in session."""
        event = SignalingEvent(
            type=message_type,
            session_id=session_id,
            peer_id=from_peer,
            data=data,
        )

        return await self._broadcast(session_id, from_peer, event)

    async def _broadcast(
        self,
        session_id: str,
        exclude_peer: str,
        event: SignalingEvent,
    ) -> int:
        """Internal broadcast implementation."""
        peers = self._sessions.get(session_id, set())
        sent = 0

        for peer_id in peers:
            if peer_id == exclude_peer:
                continue

            queue = self._peer_queues.get(peer_id)
            if queue:
                await queue.put(event)
                sent += 1

        return sent

    async def receive(
        self,
        peer_id: str,
        timeout: float = 30.0,
    ) -> Optional[SignalingEvent]:
        """Receive message for peer."""
        queue = self._peer_queues.get(peer_id)
        if not queue:
            return None

        try:
            return await asyncio.wait_for(queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def get_peers(self, session_id: str) -> List[str]:
        """Get peers in session."""
        return list(self._sessions.get(session_id, set()))

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "active_sessions": len(self._sessions),
            "total_peers": len(self._peer_queues),
            "peers_per_session": {
                sid: len(peers) for sid, peers in self._sessions.items()
            },
        }
