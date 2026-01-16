"""
Real-time Communication Protocol Module

This module defines the structured communication protocol for
voice agent real-time interactions.
"""

import asyncio
import json
import logging
import struct
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from .base import (
    Connection,
    ConnectionState,
    Message,
    MessageType,
    MessagePriority,
    Event,
    EventType,
)


logger = logging.getLogger(__name__)


class ProtocolVersion(IntEnum):
    """Protocol version numbers."""

    V1 = 1
    V2 = 2
    CURRENT = 2


class FrameType(IntEnum):
    """Binary frame types for efficient wire format."""

    # Control frames
    HANDSHAKE = 0x01
    HANDSHAKE_ACK = 0x02
    PING = 0x03
    PONG = 0x04
    CLOSE = 0x05

    # Data frames
    TEXT = 0x10
    BINARY = 0x11
    AUDIO = 0x12
    VIDEO = 0x13

    # Voice agent frames
    TRANSCRIPT = 0x20
    TTS_AUDIO = 0x21
    STT_RESULT = 0x22
    AGENT_STATE = 0x23
    FUNCTION_CALL = 0x24
    FUNCTION_RESULT = 0x25

    # Session frames
    SESSION_START = 0x30
    SESSION_UPDATE = 0x31
    SESSION_END = 0x32

    # Error frames
    ERROR = 0xF0


class CompressionType(IntEnum):
    """Compression algorithms for data frames."""

    NONE = 0
    ZLIB = 1
    LZ4 = 2


@dataclass
class ProtocolFrame:
    """
    Binary protocol frame structure.

    Wire format:
    - Header (8 bytes):
      - Version (1 byte)
      - Frame type (1 byte)
      - Flags (1 byte)
      - Reserved (1 byte)
      - Payload length (4 bytes, big-endian)
    - Payload (variable length)
    - Checksum (4 bytes, CRC32)
    """

    version: int = ProtocolVersion.CURRENT
    frame_type: FrameType = FrameType.TEXT
    flags: int = 0
    payload: bytes = b""
    sequence_number: int = 0

    # Flag bits
    FLAG_COMPRESSED = 0x01
    FLAG_ENCRYPTED = 0x02
    FLAG_ACK_REQUIRED = 0x04
    FLAG_FRAGMENTED = 0x08
    FLAG_FINAL_FRAGMENT = 0x10

    @property
    def is_compressed(self) -> bool:
        return bool(self.flags & self.FLAG_COMPRESSED)

    @property
    def is_encrypted(self) -> bool:
        return bool(self.flags & self.FLAG_ENCRYPTED)

    @property
    def requires_ack(self) -> bool:
        return bool(self.flags & self.FLAG_ACK_REQUIRED)

    @property
    def is_fragmented(self) -> bool:
        return bool(self.flags & self.FLAG_FRAGMENTED)

    @property
    def is_final_fragment(self) -> bool:
        return bool(self.flags & self.FLAG_FINAL_FRAGMENT)

    def encode(self) -> bytes:
        """Encode frame to binary format."""
        # Compress payload if needed
        payload = self.payload
        flags = self.flags

        if len(payload) > 1024 and not self.is_compressed:
            compressed = zlib.compress(payload, level=6)
            if len(compressed) < len(payload) * 0.9:
                payload = compressed
                flags |= self.FLAG_COMPRESSED

        # Build header
        header = struct.pack(
            ">BBBBII",
            self.version,
            self.frame_type,
            flags,
            0,  # Reserved
            len(payload),
            self.sequence_number,
        )

        # Calculate checksum
        checksum = zlib.crc32(header + payload) & 0xFFFFFFFF
        checksum_bytes = struct.pack(">I", checksum)

        return header + payload + checksum_bytes

    @classmethod
    def decode(cls, data: bytes) -> "ProtocolFrame":
        """Decode frame from binary format."""
        if len(data) < 16:  # Minimum frame size
            raise ValueError("Frame too small")

        # Parse header
        version, frame_type, flags, _, payload_len, seq_num = struct.unpack(
            ">BBBBII", data[:12]
        )

        # Extract payload
        payload = data[12 : 12 + payload_len]

        # Verify checksum
        expected_checksum = struct.unpack(">I", data[12 + payload_len : 16 + payload_len])[0]
        actual_checksum = zlib.crc32(data[:12] + payload) & 0xFFFFFFFF

        if expected_checksum != actual_checksum:
            raise ValueError("Checksum mismatch")

        # Decompress if needed
        if flags & cls.FLAG_COMPRESSED:
            payload = zlib.decompress(payload)

        return cls(
            version=version,
            frame_type=FrameType(frame_type),
            flags=flags & ~cls.FLAG_COMPRESSED,  # Clear compression flag
            payload=payload,
            sequence_number=seq_num,
        )


@dataclass
class HandshakeMessage:
    """Protocol handshake message."""

    version: int = ProtocolVersion.CURRENT
    client_id: str = ""
    capabilities: List[str] = field(default_factory=list)
    auth_token: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "client_id": self.client_id,
            "capabilities": self.capabilities,
            "auth_token": self.auth_token,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HandshakeMessage":
        return cls(
            version=data.get("version", ProtocolVersion.CURRENT),
            client_id=data.get("client_id", ""),
            capabilities=data.get("capabilities", []),
            auth_token=data.get("auth_token"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TranscriptMessage:
    """Real-time transcript message."""

    text: str = ""
    is_final: bool = False
    confidence: float = 0.0
    language: str = "en"
    speaker: str = ""
    timestamp_ms: int = 0
    word_timestamps: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "is_final": self.is_final,
            "confidence": self.confidence,
            "language": self.language,
            "speaker": self.speaker,
            "timestamp_ms": self.timestamp_ms,
            "word_timestamps": self.word_timestamps,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptMessage":
        return cls(
            text=data.get("text", ""),
            is_final=data.get("is_final", False),
            confidence=data.get("confidence", 0.0),
            language=data.get("language", "en"),
            speaker=data.get("speaker", ""),
            timestamp_ms=data.get("timestamp_ms", 0),
            word_timestamps=data.get("word_timestamps", []),
        )


@dataclass
class AudioMessage:
    """Audio data message."""

    audio_data: bytes = b""
    sample_rate: int = 16000
    channels: int = 1
    encoding: str = "pcm16"
    duration_ms: int = 0
    sequence: int = 0
    is_speech: bool = True
    energy_level: float = 0.0

    def to_frame(self) -> ProtocolFrame:
        """Convert to protocol frame."""
        # Create metadata header
        metadata = struct.pack(
            ">IBBHIB",
            self.sample_rate,
            self.channels,
            0 if self.encoding == "pcm16" else 1,
            self.duration_ms,
            self.sequence,
            1 if self.is_speech else 0,
        )

        return ProtocolFrame(
            frame_type=FrameType.AUDIO,
            payload=metadata + self.audio_data,
        )

    @classmethod
    def from_frame(cls, frame: ProtocolFrame) -> "AudioMessage":
        """Create from protocol frame."""
        if frame.frame_type != FrameType.AUDIO:
            raise ValueError("Not an audio frame")

        # Parse metadata header
        (
            sample_rate,
            channels,
            encoding_flag,
            duration_ms,
            sequence,
            is_speech_flag,
        ) = struct.unpack(">IBBHIB", frame.payload[:12])

        return cls(
            audio_data=frame.payload[12:],
            sample_rate=sample_rate,
            channels=channels,
            encoding="pcm16" if encoding_flag == 0 else "mulaw",
            duration_ms=duration_ms,
            sequence=sequence,
            is_speech=bool(is_speech_flag),
        )


@dataclass
class AgentStateMessage:
    """Agent state update message."""

    state: str = ""
    previous_state: str = ""
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "previous_state": self.previous_state,
            "reason": self.reason,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentStateMessage":
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.utcnow()

        return cls(
            state=data.get("state", ""),
            previous_state=data.get("previous_state", ""),
            reason=data.get("reason", ""),
            metadata=data.get("metadata", {}),
            timestamp=timestamp,
        )


@dataclass
class FunctionCallMessage:
    """Function call message for agent tool use."""

    call_id: str = ""
    function_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    timeout_ms: int = 30000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "function_name": self.function_name,
            "arguments": self.arguments,
            "timeout_ms": self.timeout_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FunctionCallMessage":
        return cls(
            call_id=data.get("call_id", ""),
            function_name=data.get("function_name", ""),
            arguments=data.get("arguments", {}),
            timeout_ms=data.get("timeout_ms", 30000),
        )


@dataclass
class FunctionResultMessage:
    """Function result message."""

    call_id: str = ""
    success: bool = True
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FunctionResultMessage":
        return cls(
            call_id=data.get("call_id", ""),
            success=data.get("success", True),
            result=data.get("result"),
            error=data.get("error"),
            execution_time_ms=data.get("execution_time_ms", 0),
        )


class ProtocolHandler(ABC):
    """Abstract protocol handler."""

    @abstractmethod
    async def on_handshake(
        self,
        connection: Connection,
        message: HandshakeMessage,
    ) -> bool:
        """Handle handshake."""
        pass

    @abstractmethod
    async def on_audio(
        self,
        connection: Connection,
        message: AudioMessage,
    ) -> Optional[AudioMessage]:
        """Handle audio message."""
        pass

    @abstractmethod
    async def on_transcript(
        self,
        connection: Connection,
        message: TranscriptMessage,
    ) -> None:
        """Handle transcript message."""
        pass

    @abstractmethod
    async def on_function_call(
        self,
        connection: Connection,
        message: FunctionCallMessage,
    ) -> FunctionResultMessage:
        """Handle function call."""
        pass

    @abstractmethod
    async def on_state_change(
        self,
        connection: Connection,
        message: AgentStateMessage,
    ) -> None:
        """Handle state change."""
        pass


class VoiceAgentProtocol:
    """
    Voice agent communication protocol.

    Handles structured communication between clients and the
    voice agent platform with support for:
    - Binary and text message encoding
    - Message sequencing and acknowledgment
    - Audio streaming with metadata
    - Real-time transcription
    - Function calls and results
    - Agent state synchronization
    """

    def __init__(
        self,
        handler: Optional[ProtocolHandler] = None,
        version: int = ProtocolVersion.CURRENT,
    ):
        """
        Initialize protocol.

        Args:
            handler: Protocol message handler
            version: Protocol version
        """
        self.handler = handler
        self.version = version

        # Sequence tracking
        self._send_sequence: int = 0
        self._recv_sequence: int = 0
        self._pending_acks: Dict[int, asyncio.Future] = {}

        # Fragment reassembly
        self._fragments: Dict[int, List[bytes]] = {}

        # Message handlers by type
        self._message_handlers: Dict[FrameType, Callable] = {
            FrameType.HANDSHAKE: self._handle_handshake,
            FrameType.PING: self._handle_ping,
            FrameType.AUDIO: self._handle_audio,
            FrameType.TRANSCRIPT: self._handle_transcript,
            FrameType.FUNCTION_CALL: self._handle_function_call,
            FrameType.FUNCTION_RESULT: self._handle_function_result,
            FrameType.AGENT_STATE: self._handle_agent_state,
        }

        # Statistics
        self._stats = {
            "frames_sent": 0,
            "frames_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "acks_pending": 0,
            "fragments_pending": 0,
        }

    async def encode_message(
        self,
        message: Union[Dict[str, Any], bytes],
        frame_type: FrameType = FrameType.TEXT,
        require_ack: bool = False,
    ) -> bytes:
        """
        Encode a message to wire format.

        Args:
            message: Message to encode
            frame_type: Frame type
            require_ack: Require acknowledgment

        Returns:
            Encoded bytes
        """
        # Convert dict to JSON bytes
        if isinstance(message, dict):
            payload = json.dumps(message).encode("utf-8")
        else:
            payload = message

        # Build flags
        flags = 0
        if require_ack:
            flags |= ProtocolFrame.FLAG_ACK_REQUIRED

        # Create frame
        self._send_sequence += 1
        frame = ProtocolFrame(
            version=self.version,
            frame_type=frame_type,
            flags=flags,
            payload=payload,
            sequence_number=self._send_sequence,
        )

        # Track pending ack
        if require_ack:
            future = asyncio.Future()
            self._pending_acks[self._send_sequence] = future
            self._stats["acks_pending"] = len(self._pending_acks)

        encoded = frame.encode()
        self._stats["frames_sent"] += 1
        self._stats["bytes_sent"] += len(encoded)

        return encoded

    async def decode_message(
        self,
        data: bytes,
        connection: Optional[Connection] = None,
    ) -> Tuple[FrameType, Any]:
        """
        Decode a message from wire format.

        Args:
            data: Encoded bytes
            connection: Associated connection

        Returns:
            Tuple of (frame_type, decoded_message)
        """
        self._stats["bytes_received"] += len(data)

        frame = ProtocolFrame.decode(data)
        self._stats["frames_received"] += 1

        # Handle fragmented frames
        if frame.is_fragmented:
            return await self._handle_fragment(frame)

        # Track sequence
        self._recv_sequence = frame.sequence_number

        # Send ack if required
        if frame.requires_ack:
            await self._send_ack(frame.sequence_number)

        # Decode payload based on frame type
        if frame.frame_type in (FrameType.TEXT, FrameType.HANDSHAKE):
            message = json.loads(frame.payload.decode("utf-8"))
        elif frame.frame_type == FrameType.AUDIO:
            message = AudioMessage.from_frame(frame)
        else:
            message = frame.payload

        # Dispatch to handler
        if connection and self.handler and frame.frame_type in self._message_handlers:
            handler_func = self._message_handlers[frame.frame_type]
            await handler_func(connection, message)

        return frame.frame_type, message

    async def _handle_fragment(
        self,
        frame: ProtocolFrame,
    ) -> Tuple[FrameType, Optional[Any]]:
        """Handle fragmented frame."""
        seq = frame.sequence_number

        if seq not in self._fragments:
            self._fragments[seq] = []

        self._fragments[seq].append(frame.payload)
        self._stats["fragments_pending"] = len(self._fragments)

        if frame.is_final_fragment:
            # Reassemble
            complete_payload = b"".join(self._fragments[seq])
            del self._fragments[seq]
            self._stats["fragments_pending"] = len(self._fragments)

            # Create complete frame
            complete_frame = ProtocolFrame(
                version=frame.version,
                frame_type=frame.frame_type,
                flags=frame.flags & ~(
                    ProtocolFrame.FLAG_FRAGMENTED |
                    ProtocolFrame.FLAG_FINAL_FRAGMENT
                ),
                payload=complete_payload,
                sequence_number=seq,
            )

            return await self.decode_message(complete_frame.encode())

        return frame.frame_type, None

    async def _send_ack(self, sequence_number: int) -> None:
        """Send acknowledgment for a frame."""
        # In practice, this would send through the connection
        logger.debug(f"ACK sent for sequence {sequence_number}")

    async def _handle_handshake(
        self,
        connection: Connection,
        message: Dict[str, Any],
    ) -> None:
        """Handle handshake message."""
        handshake = HandshakeMessage.from_dict(message)

        if self.handler:
            await self.handler.on_handshake(connection, handshake)

    async def _handle_ping(
        self,
        connection: Connection,
        message: Any,
    ) -> None:
        """Handle ping message."""
        # Respond with pong
        logger.debug(f"Ping received from {connection.id}")

    async def _handle_audio(
        self,
        connection: Connection,
        message: AudioMessage,
    ) -> None:
        """Handle audio message."""
        if self.handler:
            response = await self.handler.on_audio(connection, message)
            if response:
                # Send response audio back
                pass

    async def _handle_transcript(
        self,
        connection: Connection,
        message: Dict[str, Any],
    ) -> None:
        """Handle transcript message."""
        transcript = TranscriptMessage.from_dict(message)

        if self.handler:
            await self.handler.on_transcript(connection, transcript)

    async def _handle_function_call(
        self,
        connection: Connection,
        message: Dict[str, Any],
    ) -> None:
        """Handle function call message."""
        call = FunctionCallMessage.from_dict(message)

        if self.handler:
            result = await self.handler.on_function_call(connection, call)
            # Send result back
            logger.debug(f"Function {call.function_name} result: {result.success}")

    async def _handle_function_result(
        self,
        connection: Connection,
        message: Dict[str, Any],
    ) -> None:
        """Handle function result message."""
        result = FunctionResultMessage.from_dict(message)
        logger.debug(f"Function result received: {result.call_id}")

    async def _handle_agent_state(
        self,
        connection: Connection,
        message: Dict[str, Any],
    ) -> None:
        """Handle agent state message."""
        state = AgentStateMessage.from_dict(message)

        if self.handler:
            await self.handler.on_state_change(connection, state)

    # Convenience methods for sending specific message types

    async def send_handshake(
        self,
        client_id: str,
        capabilities: List[str],
        auth_token: Optional[str] = None,
    ) -> bytes:
        """Create handshake message."""
        message = HandshakeMessage(
            version=self.version,
            client_id=client_id,
            capabilities=capabilities,
            auth_token=auth_token,
        )

        return await self.encode_message(
            message.to_dict(),
            frame_type=FrameType.HANDSHAKE,
            require_ack=True,
        )

    async def send_audio(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        encoding: str = "pcm16",
        sequence: int = 0,
    ) -> bytes:
        """Create audio message."""
        message = AudioMessage(
            audio_data=audio_data,
            sample_rate=sample_rate,
            encoding=encoding,
            sequence=sequence,
        )

        frame = message.to_frame()
        frame.sequence_number = self._send_sequence + 1
        self._send_sequence += 1

        encoded = frame.encode()
        self._stats["frames_sent"] += 1
        self._stats["bytes_sent"] += len(encoded)

        return encoded

    async def send_transcript(
        self,
        text: str,
        is_final: bool = False,
        confidence: float = 1.0,
        speaker: str = "",
    ) -> bytes:
        """Create transcript message."""
        message = TranscriptMessage(
            text=text,
            is_final=is_final,
            confidence=confidence,
            speaker=speaker,
            timestamp_ms=int(datetime.utcnow().timestamp() * 1000),
        )

        return await self.encode_message(
            message.to_dict(),
            frame_type=FrameType.TRANSCRIPT,
        )

    async def send_function_call(
        self,
        call_id: str,
        function_name: str,
        arguments: Dict[str, Any],
        timeout_ms: int = 30000,
    ) -> bytes:
        """Create function call message."""
        message = FunctionCallMessage(
            call_id=call_id,
            function_name=function_name,
            arguments=arguments,
            timeout_ms=timeout_ms,
        )

        return await self.encode_message(
            message.to_dict(),
            frame_type=FrameType.FUNCTION_CALL,
            require_ack=True,
        )

    async def send_function_result(
        self,
        call_id: str,
        result: Any,
        success: bool = True,
        error: Optional[str] = None,
    ) -> bytes:
        """Create function result message."""
        message = FunctionResultMessage(
            call_id=call_id,
            success=success,
            result=result,
            error=error,
        )

        return await self.encode_message(
            message.to_dict(),
            frame_type=FrameType.FUNCTION_RESULT,
        )

    async def send_agent_state(
        self,
        state: str,
        previous_state: str = "",
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """Create agent state message."""
        message = AgentStateMessage(
            state=state,
            previous_state=previous_state,
            reason=reason,
            metadata=metadata or {},
        )

        return await self.encode_message(
            message.to_dict(),
            frame_type=FrameType.AGENT_STATE,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get protocol statistics."""
        return dict(self._stats)


class MessageRouter:
    """
    Routes messages between connections based on session mapping.

    Supports:
    - Direct connection routing
    - Session-based routing
    - Broadcast to groups
    - Priority-based delivery
    """

    def __init__(self):
        """Initialize router."""
        self._routes: Dict[str, str] = {}  # session_id -> connection_id
        self._connection_sessions: Dict[str, str] = {}  # connection_id -> session_id
        self._groups: Dict[str, set] = {}  # group_id -> set of connection_ids
        self._priority_queues: Dict[str, asyncio.PriorityQueue] = {}

    def register_route(
        self,
        session_id: str,
        connection_id: str,
    ) -> None:
        """Register a session-to-connection route."""
        self._routes[session_id] = connection_id
        self._connection_sessions[connection_id] = session_id

        logger.debug(f"Route registered: {session_id} -> {connection_id}")

    def unregister_route(self, session_id: str) -> None:
        """Unregister a route."""
        connection_id = self._routes.pop(session_id, None)
        if connection_id:
            self._connection_sessions.pop(connection_id, None)

        logger.debug(f"Route unregistered: {session_id}")

    def get_connection_for_session(self, session_id: str) -> Optional[str]:
        """Get connection ID for a session."""
        return self._routes.get(session_id)

    def get_session_for_connection(self, connection_id: str) -> Optional[str]:
        """Get session ID for a connection."""
        return self._connection_sessions.get(connection_id)

    def add_to_group(self, group_id: str, connection_id: str) -> None:
        """Add connection to a group."""
        if group_id not in self._groups:
            self._groups[group_id] = set()
        self._groups[group_id].add(connection_id)

    def remove_from_group(self, group_id: str, connection_id: str) -> None:
        """Remove connection from a group."""
        if group_id in self._groups:
            self._groups[group_id].discard(connection_id)

    def get_group_connections(self, group_id: str) -> List[str]:
        """Get all connections in a group."""
        return list(self._groups.get(group_id, set()))

    def get_all_routes(self) -> Dict[str, str]:
        """Get all registered routes."""
        return dict(self._routes)


def create_protocol(
    handler: Optional[ProtocolHandler] = None,
) -> VoiceAgentProtocol:
    """
    Create a voice agent protocol instance.

    Args:
        handler: Protocol handler

    Returns:
        Protocol instance
    """
    return VoiceAgentProtocol(handler=handler)


__all__ = [
    "ProtocolVersion",
    "FrameType",
    "CompressionType",
    "ProtocolFrame",
    "HandshakeMessage",
    "TranscriptMessage",
    "AudioMessage",
    "AgentStateMessage",
    "FunctionCallMessage",
    "FunctionResultMessage",
    "ProtocolHandler",
    "VoiceAgentProtocol",
    "MessageRouter",
    "create_protocol",
]
